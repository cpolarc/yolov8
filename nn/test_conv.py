"""Verification for tmp_handmake/nn/conv.py — run: python test_conv.py"""

import sys

import torch
import torch.nn as nn

import conv as hand
from ultralytics.nn.modules.conv import Conv as RefConv
from ultralytics.nn.modules.conv import Conv2 as RefConv2
from ultralytics.nn.modules.conv import autopad as ref_autopad

HandConv = hand.Conv
HandConv2 = getattr(hand, "Conv2", None) or hand.Conv2d  # tolerate either name

fails = []


def check(name, fn):
    """Run one check, record the failure instead of aborting the whole run."""
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        fails.append(name)


# ---------------------------------------------------------------- 1. autopad
print("\n[1] autopad matches upstream")
for k in (1, 3, 5, 7, (3, 3), (5, 3)):
    for d in (1, 2, 3):
        for p in (None, 0, 2):
            check(
                f"autopad(k={k}, p={p}, d={d})",
                lambda k=k, p=p, d=d: (
                    lambda a, b: (_ for _ in ()).throw(AssertionError(f"{a} != {b}")) if a != b else None
                )(hand.autopad(k, p, d), ref_autopad(k, p, d)),
            )


# ------------------------------------------------------- 2. construct + shape
print("\n[2] construction and output shape")


def shape_case(cls, kwargs, cin, hw=16):
    """Build the module and assert its output shape matches a reference nn.Conv2d+BN stack."""

    def run():
        m = cls(cin, 12, **kwargs).eval()
        x = torch.randn(2, cin, hw, hw)
        y = m(x)
        # ground truth: same conv geometry via a plain nn.Conv2d
        k, s, d = kwargs.get("k", 1), kwargs.get("s", 1), kwargs.get("d", 1)
        g = kwargs.get("g", 1)
        ref = nn.Conv2d(cin, 12, k, s, ref_autopad(k, kwargs.get("p"), d), groups=g, dilation=d, bias=False)
        assert y.shape == ref(x).shape, f"got {tuple(y.shape)}, expected {tuple(ref(x).shape)}"

    return run


for kw in [
    {},
    {"k": 3},
    {"k": 3, "s": 2},
    {"k": 5, "d": 2},  # dilation must reach BOTH autopad and nn.Conv2d
    {"k": 3, "g": 4},
    {"k": 3, "act": False},
    {"k": 3, "act": nn.ReLU()},
]:
    check(f"Conv{kw}", shape_case(HandConv, kw, 8))

for kw in [{}, {"s": 2}, {"g": 4}, {"d": 2}]:
    check(f"Conv2{kw}", shape_case(HandConv2, kw, 8))


# ---------------------------------------------- 3. numeric parity vs upstream
print("\n[3] numeric parity with ultralytics (same weights -> same output)")


def parity(hand_cls, ref_cls, kwargs):
    """Copy hand-module weights into the upstream module and compare outputs elementwise."""

    def run():
        h = hand_cls(8, 12, **kwargs).eval()
        r = ref_cls(8, 12, **kwargs).eval()
        r.load_state_dict(h.state_dict())  # also proves the parameter sets are identical
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            assert torch.allclose(h(x), r(x), atol=1e-6), "outputs differ"

    return run


for kw in [{"k": 3}, {"k": 3, "s": 2}, {"k": 5, "d": 2}, {"k": 3, "g": 4}]:
    check(f"Conv{kw} vs upstream", parity(HandConv, RefConv, kw))
for kw in [{}, {"s": 2}, {"g": 4}]:
    check(f"Conv2{kw} vs upstream", parity(HandConv2, RefConv2, kw))


# ------------------------------------------------------- 4. fusion invariance
print("\n[4] fuse_convs is numerically transparent")


def fuse_case(kwargs):
    """fuse_convs() folds the 1x1 branch into the 3x3 — output must not change."""

    def run():
        m = HandConv2(8, 12, **kwargs)
        # randomise BN so an identity-BN doesn't hide a bug
        nn.init.uniform_(m.bn.weight, 0.5, 1.5)
        nn.init.uniform_(m.bn.bias, -0.5, 0.5)
        m.bn.running_mean.uniform_(-1, 1)
        m.bn.running_var.uniform_(0.5, 2.0)
        m.eval()  # CRITICAL: train mode uses batch stats and hides fusion errors
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            before = m(x)
            m.fuse_convs()
            after = m(x)
        assert not hasattr(m, "cv2"), "cv2 branch not removed"
        d = (before - after).abs().max().item()
        assert torch.allclose(before, after, atol=1e-5), f"max abs diff {d:.3e}"

    return run


for kw in [{}, {"s": 2}, {"g": 4}, {"k": 3, "d": 2}]:
    check(f"fuse{kw}", fuse_case(kw))


# ---------------------------------------------------------- 5. known sharp edges
print("\n[5] sharp edges — these two FAIL upstream too, so they document a contract, not a bug")

check(
    "default_act spelled correctly (external override works)",
    lambda: (_ for _ in ()).throw(AssertionError("no `default_act` attribute — external act override silently no-ops"))
    if not hasattr(HandConv, "default_act")
    else None,
)


def explicit_padding():
    """With an explicit p, the 1x1 branch reuses it and the shapes stop matching."""
    m = HandConv2(8, 12, k=3, p=1).eval()
    m(torch.randn(2, 8, 16, 16))


check("Conv2 with explicit p=1", explicit_padding)


def even_kernel():
    """Even k: autopad over-pads the main branch, so forward breaks before fuse is ever reached.

    autopad(4) == 2 gives a 17x17 main branch against a 16x16 1x1 branch. Upstream Conv2 behaves
    identically. The off-centre-fold concern for even kernels is therefore unreachable in practice.
    """
    m = HandConv2(8, 12, k=4).eval()
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        before = m(x)
        m.fuse_convs()
        after = m(x)
    assert torch.allclose(before, after, atol=1e-5), "even-k fusion is not transparent"


check("even kernel k=4 (forward breaks first)", even_kernel)


def same_failure(hand_fn, ref_fn, label):
    """A sharp edge only excuses the hand implementation if upstream fails the same way."""

    def outcome(fn):
        try:
            fn()
            return None
        except Exception as e:
            return type(e).__name__

    def run():
        h, r = outcome(hand_fn), outcome(ref_fn)
        assert h == r, f"hand={h} but upstream={r} — divergence, not a shared contract"
        assert h is not None, "both succeeded; this is no longer a sharp edge"

    check(f"{label}: upstream fails identically", run)


same_failure(
    explicit_padding,
    lambda: RefConv2(8, 12, k=3, p=1).eval()(torch.randn(2, 8, 16, 16)),
    "explicit p=1",
)
same_failure(
    even_kernel,
    lambda: RefConv2(8, 12, k=4).eval()(torch.randn(2, 8, 16, 16)),
    "even kernel k=4",
)


print(f"\n{'=' * 60}")
print(f"{len(fails)} failing check(s)" + (f": {fails}" if fails else " — all good"))
sys.exit(1 if fails else 0)
