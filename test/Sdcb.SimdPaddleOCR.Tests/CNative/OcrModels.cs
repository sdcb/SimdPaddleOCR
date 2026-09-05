namespace LwPpocrCSharp;

sealed class OcrResponse
{
    public bool ok { get; set; }
    public int api_version { get; set; }
    public string request_id { get; set; } = "";
    public int image_width { get; set; }
    public int image_height { get; set; }
    public int detected_count { get; set; }
    public List<PaddleOcrLine> result { get; set; } = [];
}

sealed class PaddleOcrLine
{
    public string text { get; set; } = "";
    public float score { get; set; }
    public float det_score { get; set; }
    public int cls_label { get; set; }
    public float cls_score { get; set; }
    public int rotation { get; set; }
    public float x1 { get; set; }
    public float y1 { get; set; }
    public float x2 { get; set; }
    public float y2 { get; set; }
    public float x3 { get; set; }
    public float y3 { get; set; }
    public float x4 { get; set; }
    public float y4 { get; set; }
}

sealed class DecodedBgrImage
{
    public byte[] Pixels = [];
    public int Width;
    public int Height;
    public int Stride;
}
