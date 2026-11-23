"""工具包。

导入本包即完成所有工具的注册——这是刻意的：

    @register 装饰器只有在模块被 import 时才会执行。如果业务代码
    只 `from .tools import registry`，那些定义了工具的模块根本不会
    被加载，registry 就是空的。表现是「工具调用失败：未知工具」，
    而且不会报导入错误，很容易查半天。

    所以在这里集中 import 一遍，把「注册」和「使用」解耦：
    任何地方 import 到这个包，工具就都在。

    注意导入顺序：registry 必须最先，其它模块依赖它。
"""

from . import registry  # noqa: F401  必须最先
from . import diagnostic, kb_search, metrics, ticket  # noqa: F401,E402

__all__ = ["registry", "kb_search", "metrics", "diagnostic", "ticket"]
