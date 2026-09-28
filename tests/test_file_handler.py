"""文件处理：MD5 分片计算、目录扫描的边界返回。"""
import hashlib
from pathlib import Path

from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type


# ----------------------------------------------------------------------
# MD5
# ----------------------------------------------------------------------

def test_md5_matches_hashlib(tmp_path):
    target = tmp_path / "a.txt"
    payload = "图书馆知识库内容" * 100
    target.write_text(payload, encoding="utf-8")

    expected = hashlib.md5(payload.encode("utf-8")).hexdigest()

    assert get_file_md5_hex(str(target)) == expected


def test_md5_is_content_addressed(tmp_path):
    """同内容不同文件名 → 同 MD5；改一个字符 → 不同 MD5。

    这是增量加载去重的全部依据。
    """
    (tmp_path / "one.txt").write_text("相同内容", encoding="utf-8")
    (tmp_path / "two.txt").write_text("相同内容", encoding="utf-8")
    (tmp_path / "three.txt").write_text("相同内文", encoding="utf-8")

    a = get_file_md5_hex(str(tmp_path / "one.txt"))
    b = get_file_md5_hex(str(tmp_path / "two.txt"))
    c = get_file_md5_hex(str(tmp_path / "three.txt"))

    assert a == b
    assert a != c


def test_md5_handles_large_file_in_chunks(tmp_path):
    """4KB 分片流式读取 —— 内容跨分片边界也不能算错。"""
    target = tmp_path / "big.bin"
    payload = bytes(range(256)) * 100        # 25600 字节，跨 6 个分片
    target.write_bytes(payload)

    assert get_file_md5_hex(str(target)) == hashlib.md5(payload).hexdigest()


def test_md5_of_missing_path_returns_none(tmp_path):
    assert get_file_md5_hex(str(tmp_path / "不存在.txt")) is None


def test_md5_of_directory_returns_none(tmp_path):
    """目录不是文件，必须返回 None 而不是抛异常。"""
    assert get_file_md5_hex(str(tmp_path)) is None


# ----------------------------------------------------------------------
# 目录扫描
# ----------------------------------------------------------------------

def test_listdir_filters_by_suffix(tmp_path):
    for name in ("a.txt", "b.txt", "c.pdf", "d.md", "e.png"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    found = listdir_with_allowed_type(str(tmp_path), ("txt", "pdf"))

    names = sorted(Path(f).name for f in found)
    assert names == ["a.txt", "b.txt", "c.pdf"]


def test_listdir_returns_full_paths(tmp_path):
    """返回的是可直接读取的完整路径，不是文件名 —— 下游要拿它算 MD5。"""
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")

    (found,) = listdir_with_allowed_type(str(tmp_path), ("txt",))

    assert Path(found).is_file()
    assert Path(found).is_absolute()


def test_listdir_on_non_directory_returns_empty_tuple(tmp_path):
    """这里曾是个静默错误。

    原实现 ``return allowed_types`` 会返回 ``("txt","pdf")`` 这个**后缀字符串元组**，
    与正常路径返回的**文件路径元组**长度和类型都相同、语义却完全相反；调用方会拿
    字符串 ``"txt"`` 去算 MD5。类型注解捕获不到这种错误，所以用测试钉死。
    """
    result = listdir_with_allowed_type(str(tmp_path / "不存在"), ("txt", "pdf"))

    assert result == ()
    assert result != ("txt", "pdf"), "必须返回空元组，不能返回后缀串"


def test_listdir_on_empty_directory(tmp_path):
    assert listdir_with_allowed_type(str(tmp_path), ("txt",)) == ()
