"""口令哈希工具：加盐 PBKDF2-SHA256，仅依赖标准库。

不存明文口令，也不使用可逆加密。存储格式为 ``<salt_hex>$<hash_hex>``，
校验时从存储串中取回同一份盐重算，再用常量时间比较，避免时序侧信道。
"""
import hashlib
import hmac
import secrets

# 迭代次数：兼顾安全与登录响应速度（约几十毫秒量级）
_ITERATIONS = 200_000
_SALT_BYTES = 16
_ALGORITHM = "sha256"


def _derive(password: str, salt: bytes, iterations: int = _ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac(_ALGORITHM, password.encode("utf-8"), salt, iterations)


def hash_password(password: str) -> str:
    """生成口令的加盐哈希串，格式 ``salt$hash``。"""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _derive(password, salt)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验口令是否与存储的哈希串匹配。

    存储串格式非法时返回 False 而非抛异常——调用方（登录流程）只关心
    「能否通过」，不该因一条脏数据导致整个页面崩溃。
    """
    if not password or not stored or "$" not in stored:
        return False

    salt_hex, hash_hex = stored.split("$", 1)
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False

    return hmac.compare_digest(_derive(password, salt), expected)
