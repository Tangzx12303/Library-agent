"""口令哈希：PBKDF2-HMAC-SHA256 + 随机盐 + 常量时间比较。"""
from utils.security import hash_password, verify_password


def test_roundtrip():
    stored = hash_password("123456")

    assert verify_password("123456", stored)


def test_wrong_password_rejected():
    stored = hash_password("123456")

    assert not verify_password("123457", stored)
    assert not verify_password("", stored)


def test_same_password_yields_different_hashes():
    """每口令独立随机盐：相同口令不得产生相同哈希（防彩虹表）。"""
    a, b = hash_password("123456"), hash_password("123456")

    assert a != b
    assert verify_password("123456", a) and verify_password("123456", b)


def test_stored_format_is_salt_dollar_hash():
    salt_hex, hash_hex = hash_password("123456").split("$", 1)

    assert len(salt_hex) == 32          # 16 字节盐 → 32 个十六进制字符
    assert len(hash_hex) == 64          # SHA-256 → 32 字节 → 64 个十六进制字符
    int(salt_hex, 16) and int(hash_hex, 16)   # 必须都是合法十六进制


def test_tampered_hash_is_rejected():
    """改掉哈希的任意一个字符都必须校验失败。"""
    salt_hex, hash_hex = hash_password("123456").split("$", 1)
    flipped = "0" if hash_hex[0] != "0" else "1"
    tampered = f"{salt_hex}${flipped}{hash_hex[1:]}"

    assert not verify_password("123456", tampered)


def test_dirty_stored_value_returns_false_instead_of_raising():
    """脏数据（缺分隔符 / 非十六进制 / 空）只该返回 False，不该抛异常 ——
    登录流程不该因库里一条脏记录就整页崩掉。"""
    for dirty in ("", "no-separator", "zz$zz", "$", "abc$"):
        assert verify_password("123456", dirty) is False


def test_none_inputs_are_safe():
    assert verify_password(None, "x$y") is False
    assert verify_password("123456", None) is False
