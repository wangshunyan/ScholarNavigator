# Full1000 来源容量与确定性节流

`formal_provider_pacing_v1` 只消费已冻结
`formal_network_request_manifest_v1` 的 9,640 个 request intent。它不会重新规划 query、
改写参数、增删来源、提前物化响应依赖 cursor，也不会调用 adapter。每个 intent 的
opaque identity、request-spec 摘要、cache identity、shard 和两次 HTTP attempt 理论上限
保持不变。

## 容量声明

正式容量声明按来源分别登记：

- 每秒和每分钟请求上限；
- 最大并发、burst 与冷却窗口；
- 声明版本、有效起止边界和外部来源证明。

仓库没有可核验的四源容量声明，因此这些字段全部为 `not_available`。历史 429/503、
已有 connector 延迟和合成运行速度均不能用于推测容量；`audit-readiness` 固定返回 exit 3，
正式启动保持阻断。测试内的容量 profile 显式标记 `synthetic`，不能进入 launch
authorization、readiness 的真实容量槽位或资源成本声明。

## 调度和退避语义

逻辑时钟以一个声明容量秒为步长。每个来源有独立 token bucket、分钟滑动窗口和并发槽，
另受冻结 scheduler 的全局并发 12 限制。所有 9,640 个首次 intent 被 admission 后，
pagination/retry 才按稳定父请求顺序进入 continuation 队列；慢源、429 或 retry 不得通过
选择性跳过改变请求人口。

429 的有效 `Retry-After` 优先于声明冷却窗口；503/timeout 使用声明冷却窗口。它们只消费
intent 已有的 attempt 上限，不创建额外预算。pause/cancel 后停止新 admission，在途操作
结束并写入唯一账本记录；resume 恢复 token、分钟窗口、cooldown、失败计数、source cursor
和已 admission identity，禁止重复请求或清零配额状态。

## 离线门禁

```bash
PYTHONPATH=src python scripts/check_formal_provider_pacing.py verify-policy
PYTHONPATH=src python scripts/check_formal_provider_pacing.py simulate-capacity
PYTHONPATH=src python scripts/check_formal_provider_pacing.py verify-resume
PYTHONPATH=src python scripts/check_formal_provider_pacing.py audit-readiness
```

退出码为：

- `0`：节流控制或合成演练通过；
- `2`：容量、身份、并发、预算或恢复违规；
- `3`：缺少真实来源容量声明；
- `4`：命令用法错误。

合成矩阵覆盖均衡容量、单源低配额、burst 耗尽、`Retry-After`、持续 429、503/timeout
抖动、动态降额、暂停恢复和 shard 异步完成，并明确保留过期声明和未知容量两个阻断场景。
其中的逻辑步、等待步和并发峰值仅用于控制正确性，不是真实吞吐性能，也不证明检索相关性、
Precision、Recall 或官方成绩。
