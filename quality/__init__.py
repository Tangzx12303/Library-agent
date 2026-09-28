"""对话质量评测。

分三层，依赖方向单向（上层可以依赖下层，下层不知道上层存在）：

- **纯逻辑层**：``trace`` / ``metrics`` / ``scenario`` / ``recommendation`` /
  ``checks`` / ``evaluator`` —— 不碰数据库、不碰 Streamlit，因此可以被 pytest
  直接测。
- **存储层**：``store`` —— 唯一会写数据库的地方，且**永不抛异常**（质检失败
  不该影响读者的对话）。
- **展示层**：``aggregate`` / ``render`` —— 纯函数，把记录变成图表数据和文案，
  同样不含 Streamlit，所以也能测。``app.py`` 只负责把它们接到界面上。

包名用 ``quality`` 而不是 ``eval``：``eval`` 是 Python 内置函数名，
``import eval`` 会在导入模块的命名空间里把它遮蔽掉。
"""
