# 示例：就地贴识别结果

目标不是翻译，是：**盖掉原字，按检测框的方向和高度把识别结果写回去**，字色尽量跟原字。不描检测多边形。算法不要 sunk 进 `Sdcb.SimdPaddleOCR` 核心。

网页参考是 [`overlay.js`](../examples/SimdCodecs.AspNetCore/wwwroot/overlay.js)。三个桌面示例已落地，各项目一份原生 `OcrOverlay.cs`，不要再抄旧的绿框红字。

```javascript
OcrOverlay.draw(ctx, image, lines, { showOriginal: false });
// lines: [{ text, box: [[x,y],[x,y],[x,y],[x,y]] }]
```

## 先不要做的

- 不要沿框法向去采邻域、不要做 Telea / inpaint。密表行距小，法向 ±3px 就会采到上一行数字。
- 不要用整图一个字号，红字画在框角上方。
- 不要用 `PaddleOcrLine.AppliedRotationDegrees` 当倾角。那是 CLS 的 0/180。
- 不要竖着一个字母一个字母排 `HELLO`。90° 是整行旋转，字母躺着。
- 不要加填底/对照图开关。UI：**跑完默认就地替换**；可选「显示原图」。OCR 成功后取消勾选。
- WPF 不要用 `Cv2.PutText` 画中文。

## 实现落点

| 文件 | 画布 |
| --- | --- |
| [`overlay.js`](../examples/SimdCodecs.AspNetCore/wwwroot/overlay.js) | Canvas 2D |
| [`SystemDrawing.WinForms/OcrOverlay.cs`](../examples/SystemDrawing.WinForms/OcrOverlay.cs) | GDI+ `FillPolygon` / `DrawString` |
| [`SkiaSharp.Avalonia/OcrOverlay.cs`](../examples/SkiaSharp.Avalonia/OcrOverlay.cs) | Skia `DrawPath` / `DrawText` |
| [`OpenCvSharp5.Wpf/OcrOverlay.cs`](../examples/OpenCvSharp5.Wpf/OcrOverlay.cs) | GDI+（OpenCV 只负责读图和推理） |
| [`app.js`](../examples/SimdCodecs.AspNetCore/wwwroot/app.js) | 上传 / API / `#showOriginal` |

页面只调用 overlay，桌面窗口只缓存原图 + 贴字图并切预览。

## 每条 `PaddleOcrLine` 做什么

```
p0,p1,p2,p3 = Box 四点（通常 p0→p1 是长边）
geom = 长边方向、中心、长、高、倾角
框内像素：量化众数 → 纯色背景；与底差得大的像素 → 字色
整框铺该纯色
字号二分，落入 0.94*长 × 0.9*高
save → 中心 → rotate → 字色居中写 text → restore
```

### 几何

- `len = |p1-p0|`，`ht = |p3-p0|`。若 `ht > len`，长边改 `p0→p3`。
- `angle = atan2(along.y, along.x)`。中心四点平均。桌面旋转 API 用角度制。
- `AppliedRotationDegrees` 贴字不用。

### 纯色背景（框内众数）

扫四边形内部像素，按 **R,G,B** 各右移 4 位做桶（GDI+ / Skia 内存是 BGRA，取样时不要把第一字节当 R）。计数最多的桶的均值当背景。

密字框里黑墨可能赢过众数：若第一名占比 < 55% 且很暗、第二名明显是浅底（亮度 > 140），改用第二名。

### 字色

和背景 RGB 距离 ≥ 16 的当墨水；再取距离 ≥ 中位数的那一半，RGB 各取中位数。采不到就按底色亮度退回近黑/近白。

### 写字

- `"Microsoft YaHei UI"`，粗体。
- 画在框中心。Canvas 用 `center`/`middle`；GDI+ 用 `StringFormat` 居中；Skia 的 Y 是基线，要用 font metrics 抬到垂直居中。
- 不要描绿框。

## 桌面注意

- 必须同时留原图和贴字图。勾选「显示原图」只切预览，不要再跑推理。
- Avalonia 必须 `SKBitmap.Copy()` 再画，不能画在 OCR 那张图上。
- WinForms 双 TFM（含 net48），overlay 不要用 net10 才有的 API。
- 后画的框会采到先铺的色块，和网页一样，接受。

## 验收

1. 微信绿气泡：原字被绿底盖住，识别字在框里，字色接近原黑字。
2. 密表：格子是干净色块，不能把上一行数字拖成残影。
3. 斜字、整行 90°：字顺着框；90° 拉丁词字母躺着。
4. 「显示原图」勾上只见原图。
5. 不要为可视化改推理结果。

## 已知边界

- 渐变底会变成一条单色，这是有意的。
- 识别比原句长很多时字号会变小。不要自动换行。
- Web 示例用 SimdCodecs 直接解码成紧密排列的 BGR24，无需拷贝。桌面 `LockBits` / `Mat` 不受影响。
