# test_algorithm 插件

## 描述
test_algorithm 检测算法插件

## 版本
- 插件版本: 1.0.0
- SDK版本: 1.0.0

## 配置参数
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| threshold | number | 0.5 | 检测阈值 (0.0-1.0) |

## 使用方法
1. 将插件目录放置在 `plugins/` 文件夹下
2. 重启 AI Box 服务
3. 插件将自动加载

## 开发指南
参考 `algorithm.py` 中的实现

## API 参考
- `get_metadata()`: 返回算法元数据
- `initialize(config)`: 初始化算法
- `process(frame, context)`: 处理帧并返回结果
- `cleanup()`: 清理资源
