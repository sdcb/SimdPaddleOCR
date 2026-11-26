# Docker 部署与 HTTP API 文档

`sdcb/simdpaddleocr` 是基于 `examples/SimdCodecs.AspNetCore` 构建的官方容器镜像：

- 纯 C# 零原生依赖（`Sdcb.SimdPaddleOCR` + `Sdcb.SimdCodecs`），基础镜像为 `noble-chiseled`
- 内置 PP-OCRv6 中文 `tiny` / `small` / `medium` 三个模型（含文本行方向分类 CLS），开箱即用
- 同时提供 `linux/amd64` 与 `linux/arm64` 多架构镜像
- 自带 Web 体验界面、REST API 与 Scalar 交互式 OpenAPI 文档

## 快速启动

```bash
docker run --rm -p 8080:8080 sdcb/simdpaddleocr:latest
```

启动后访问：

- Web 体验界面：`http://localhost:8080/`
- 交互式 API 文档（Scalar）：`http://localhost:8080/scalar`
- OpenAPI 规范：`http://localhost:8080/openapi/v1.json`
- 健康检查：`http://localhost:8080/health`

## Docker Compose 示例

```yaml
services:
  ocr:
    image: sdcb/simdpaddleocr:latest
    ports:
      - "8080:8080"
    environment:
      OCR__DefaultModel: tiny
      OCR__PreloadModels: tiny
      OCR__MaxUploadBytes: 20971520
      OCR__MaxImagePixels: 40000000
      # OCR__ApiKey: your-secret-key
    restart: unless-stopped
```

## 镜像标签策略

| 标签 | 说明 |
| --- | --- |
| `latest` | 最新正式版（打正式 `v*` Git Tag 时更新） |
| `3.0.0` / `3.0` | 对应正式版本号（不含预发布后缀） |
| `r-<run_number>` | 每次 `main` 构建或发布产出的不可变构建号镜像，适合固定版本验证 |

## 环境变量配置

所有参数均通过标准 ASP.NET Core 配置键 `OCR__*` 传入：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OCR__DefaultModel` | `tiny` | 请求未指定 `model` 时的默认模型（`tiny` / `small` / `medium`） |
| `OCR__PreloadModels` | 空（按需懒加载） | 启动时预加载的模型列表，逗号分隔，如 `tiny`、`tiny,small` 或 `all` |
| `OCR__MaxUploadBytes` | `20971520` (20 MB) | 单次上传图片最大字节数 |
| `OCR__MaxImagePixels` | `40000000` (4000 万) | 单张图片最大像素数（`width * height`） |
| `OCR__ApiKey` | 空（不鉴权） | 设置后，`/api/*` 接口必须携带 `X-API-Key: <key>` 或 `Authorization: Bearer <key>`；`/health` 不受影响 |
| `OCR__Backend` | `Auto` | 推理后端（`Auto` / `Cpu` / `Vulkan`） |
| `OCR__LineWorkerCount` | `0`（自动） | 文本行 CLS/REC 并发 worker 路数（`0` 表示 `min(ProcessorCount, 4)`） |
| `OCR__DetIntraOpThreads` | `0`（自动） | 检测模型图内算子线程数 |
| `OCR__UseDirectionClassification` | `true` | 是否启用 180° 文本行方向分类 |

## HTTP API 参考

### `GET /health`

返回服务健康状态与当前已加载到内存的模型列表：

```json
{
  "status": "ok",
  "defaultModel": "tiny",
  "loadedModels": ["tiny"],
  "buildConfiguration": "Release"
}
```

### `GET /api/ocr/models`

列出可用模型与默认模型：

```json
{
  "models": ["tiny", "small", "medium"],
  "default": "tiny"
}
```

### `POST /api/ocr`

`multipart/form-data` 上传图片并返回识别结果：

| 表单字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 是 | 图片文件（JPEG / PNG / WebP / BMP / GIF / TIFF 等，自动应用 EXIF 方向） |
| `model` | string | 否 | `tiny` / `small` / `medium`，省略则使用 `OCR__DefaultModel` |
| `characters` | boolean | 否 | 为 `true` 时每行额外返回逐字坐标框 `characters`，默认 `false` |

#### curl 示例

```bash
curl -X POST http://localhost:8080/api/ocr \
  -F "file=@sample.jpg" \
  -F "model=tiny" \
  -F "characters=true"
```

启用 `OCR__ApiKey` 时附带请求头：

```bash
curl -X POST http://localhost:8080/api/ocr \
  -H "X-API-Key: your-secret-key" \
  -F "file=@sample.jpg"
```

#### Python 示例

```python
import requests

with open("sample.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8080/api/ocr",
        files={"file": f},
        data={"model": "tiny", "characters": "false"},
        timeout=30,
    )
resp.raise_for_status()
data = resp.json()
print(data["text"])
```

#### C# 示例

```csharp
using var http = new HttpClient();
using var form = new MultipartFormDataContent();
form.Add(new ByteArrayContent(await File.ReadAllBytesAsync("sample.jpg")), "file", "sample.jpg");
form.Add(new StringContent("tiny"), "model");

using var resp = await http.PostAsync("http://localhost:8080/api/ocr", form);
resp.EnsureSuccessStatusCode();
Console.WriteLine(await resp.Content.ReadAsStringAsync());
```

## 本地构建镜像（无需 Dockerfile）

本项目直接通过 .NET 10 SDK 的 `PublishContainer` 构建镜像：

```bash
# 构建并加载到本地 Docker daemon
dotnet publish examples/SimdCodecs.AspNetCore/SimdCodecs.AspNetCore.csproj \
  -c Release \
  /t:PublishContainer \
  -p:ContainerImageTag=local
```
