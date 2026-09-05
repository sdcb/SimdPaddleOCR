using System.Buffers;

using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Managed equivalent of PaddleOCR's DBPostProcess (quad mode).
/// </summary>
internal static class DbPostprocess
{
    internal readonly record struct Point(float X, float Y);
    private struct Rectangle
    {
        public float Ux, Uy, Vx, Vy, MinU, MaxU, MinV, MaxV;
    }

    public static PaddleOcrDetectionBox[] Run(ReadOnlySpan<float> prediction, int mapWidth, int mapHeight,
        PaddleOcrDetectorOptions options, int sourceWidth, int sourceHeight, float widthRatio,
        float heightRatio, Workspace scratch)
    {
        if (prediction.Length != checked(mapWidth * mapHeight))
            throw new ArgumentException("Invalid detector output size.");
        int pixels = checked(mapWidth * mapHeight);
        scratch.Ensure(pixels);
        byte[] bitmap = scratch.Bitmap;
        byte[] visited = scratch.Visited;
        int[] queue = scratch.Queue;
        Point[] boundaryScratch = scratch.Boundary;
        Point[] hullScratch = scratch.Hull;
        byte[] backgroundVisited = scratch.BackgroundVisited;
        Threshold.Binarize(prediction, bitmap, pixels, options.BitmapThreshold);
        if (options.UseDilation) Dilate2x2(bitmap, scratch.Dilated, mapWidth, mapHeight, pixels);
        Array.Clear(visited, 0, pixels);
        Array.Clear(backgroundVisited, 0, pixels);
        List<PaddleOcrDetectionBox> found = [with(Math.Min(options.MaxCandidates, 256))];
        Span<Point> corners = stackalloc Point[4];
        Span<Point> miniBoxScratch = stackalloc Point[4];
        Span<Point> expandedMiniBoxScratch = stackalloc Point[4];
        Span<Point> remappedCornersScratch = stackalloc Point[4];
        int candidateCount = 0;
        for (int start = 0; start < pixels && candidateCount < options.MaxCandidates; start++)
        {
            if (bitmap[start] == 0 || visited[start] != 0) continue;
            int boundaryCount = FillForeground(start, bitmap, visited, queue,
                boundaryScratch, mapWidth, mapHeight);
            if (boundaryCount < 3) continue;
            candidateCount++;
            if (TryBuildDetection(boundaryScratch, boundaryCount, prediction,
                mapWidth, mapHeight, options, sourceWidth, sourceHeight, hullScratch,
                corners, miniBoxScratch, expandedMiniBoxScratch, remappedCornersScratch,
                out PaddleOcrDetectionBox detection))
                found.Add(detection);
        }

        // RETR_LIST also returns contours for enclosed background regions.
        // A component-only connected-component pass misses those holes.
        // First flood-fill the exterior background from the image border;
        // this is equivalent to discarding components that touch a border,
        // but avoids collecting/checking boundary neighbors for every pixel
        // in the usually very large exterior component.  The remaining
        // zero components are holes and are traced with 4-connectivity (the
        // complementary topology of OpenCV's 8-connected foreground
        // contours), preserving the official hollow-glyph behavior.
        for (int x = 0; x < mapWidth; x++)
        {
            FillBackground(x, bitmap, backgroundVisited, queue, null,
                mapWidth, mapHeight, out _);
            if (mapHeight > 1) FillBackground((mapHeight - 1) * mapWidth + x,
                bitmap, backgroundVisited, queue, null, mapWidth, mapHeight, out _);
        }
        for (int y = 1; y + 1 < mapHeight; y++)
        {
            FillBackground(y * mapWidth, bitmap, backgroundVisited, queue, null,
                mapWidth, mapHeight, out _);
            if (mapWidth > 1) FillBackground(y * mapWidth + mapWidth - 1,
                bitmap, backgroundVisited, queue, null, mapWidth, mapHeight, out _);
        }

        for (int start = 0; start < pixels && candidateCount < options.MaxCandidates; start++)
        {
            if (bitmap[start] != 0 || backgroundVisited[start] != 0) continue;
            FillBackground(start, bitmap, backgroundVisited, queue, boundaryScratch,
                mapWidth, mapHeight, out int boundaryCount);
            if (boundaryCount < 3) continue;
            candidateCount++;
            if (TryBuildDetection(boundaryScratch, boundaryCount, prediction,
                mapWidth, mapHeight, options, sourceWidth, sourceHeight, hullScratch,
                corners, miniBoxScratch, expandedMiniBoxScratch, remappedCornersScratch,
                out PaddleOcrDetectionBox detection))
                found.Add(detection);
        }
        SortReadingOrder(found);
        return [.. found];
    }

    /// <summary>
    /// Grow-only DB scratch sized to LimitSideLength² so every image reuses one
    /// LOH size instead of pinning a unique bucket per map shape.
    /// </summary>
    internal sealed class Workspace
    {
        public byte[] Bitmap = [];
        public byte[] Visited = [];
        public byte[] BackgroundVisited = [];
        public byte[] Dilated = [];
        public int[] Queue = [];
        public Point[] Boundary = [];
        public Point[] Hull = [];

        public void Ensure(int pixels)
        {
            if (pixels <= 0) return;
            if (Bitmap.Length < pixels) Bitmap = new byte[pixels];
            if (Visited.Length < pixels) Visited = new byte[pixels];
            if (BackgroundVisited.Length < pixels) BackgroundVisited = new byte[pixels];
            if (Dilated.Length < pixels) Dilated = new byte[pixels];
            int queue = checked(3 * (pixels / 2 + 2));
            if (Queue.Length < queue) Queue = new int[queue];
            if (Boundary.Length < pixels) Boundary = new Point[pixels];
            int hull = checked(pixels * 2);
            if (Hull.Length < hull) Hull = new Point[hull];
        }
    }

    // Scanline flood fill over the 8-connected foreground component containing
    // `start`.  Produces the same visited set and the same boundary-point set
    // as the previous per-pixel BFS (each boundary pixel emitted exactly once;
    // the convex hull consumer is order-insensitive).
    private static int FillForeground(int start, byte[] bitmap, byte[] visited, int[] stack,
        Point[] boundary, int mapWidth, int mapHeight)
    {
        int boundaryCount = 0, top = 0;
        int seedY = start / mapWidth, seedX = start % mapWidth, seedRow = seedY * mapWidth;
        int seedLeft = seedX, seedRight = seedX;
        while (seedLeft > 0 && bitmap[seedRow + seedLeft - 1] != 0 && visited[seedRow + seedLeft - 1] == 0) seedLeft--;
        while (seedRight + 1 < mapWidth && bitmap[seedRow + seedRight + 1] != 0 && visited[seedRow + seedRight + 1] == 0) seedRight++;
        ArrayCompat.Fill(visited, (byte)1, seedRow + seedLeft, seedRight - seedLeft + 1);
        stack[top++] = seedY; stack[top++] = seedLeft; stack[top++] = seedRight;
        while (top > 0)
        {
            int right = stack[--top], left = stack[--top], y = stack[--top];
            for (int x = left; x <= right; x++)
                if (IsBoundary(bitmap, mapWidth, mapHeight, x, y) && boundaryCount < boundary.Length)
                    boundary[boundaryCount++] = new Point(x, y);
            int scanFrom = Math.Max(0, left - 1), scanTo = Math.Min(mapWidth - 1, right + 1);
            for (int direction = -1; direction <= 1; direction += 2)
            {
                int neighborY = y + direction;
                if ((uint)neighborY >= (uint)mapHeight) continue;
                int row = neighborY * mapWidth, x = scanFrom;
                while (x <= scanTo)
                {
                    if (bitmap[row + x] != 0 && visited[row + x] == 0)
                    {
                        int runLeft = x, runRight = x;
                        while (runLeft > 0 && bitmap[row + runLeft - 1] != 0 && visited[row + runLeft - 1] == 0) runLeft--;
                        while (runRight + 1 < mapWidth && bitmap[row + runRight + 1] != 0 && visited[row + runRight + 1] == 0) runRight++;
                        ArrayCompat.Fill(visited, (byte)1, row + runLeft, runRight - runLeft + 1);
                        stack[top++] = neighborY; stack[top++] = runLeft; stack[top++] = runRight;
                        x = runRight + 2;
                    }
                    else x++;
                }
            }
        }
        return boundaryCount;
    }

    // Scanline flood fill over the 4-connected background component containing
    // `seed`.  When `boundary` is non-null this reproduces the hole-tracing
    // contract of the previous BFS: every (background pixel, foreground
    // neighbor) adjacency contributes one boundary entry at the background
    // pixel's coordinates, including duplicates.
    private static void FillBackground(int seed, byte[] bitmap, byte[] visited, int[] stack,
        Point[]? boundary, int mapWidth, int mapHeight, out int boundaryCount)
    {
        boundaryCount = 0;
        if (bitmap[seed] != 0 || visited[seed] != 0) return;
        int top = 0;
        int seedY = seed / mapWidth, seedX = seed % mapWidth, seedRow = seedY * mapWidth;
        int seedLeft = seedX, seedRight = seedX;
        while (seedLeft > 0 && bitmap[seedRow + seedLeft - 1] == 0 && visited[seedRow + seedLeft - 1] == 0) seedLeft--;
        while (seedRight + 1 < mapWidth && bitmap[seedRow + seedRight + 1] == 0 && visited[seedRow + seedRight + 1] == 0) seedRight++;
        ArrayCompat.Fill(visited, (byte)1, seedRow + seedLeft, seedRight - seedLeft + 1);
        stack[top++] = seedY; stack[top++] = seedLeft; stack[top++] = seedRight;
        while (top > 0)
        {
            int right = stack[--top], left = stack[--top], y = stack[--top];
            if (boundary is not null)
            {
                int row = y * mapWidth;
                if (left > 0 && bitmap[row + left - 1] != 0 && boundaryCount < boundary.Length)
                    boundary[boundaryCount++] = new Point(left, y);
                if (right + 1 < mapWidth && bitmap[row + right + 1] != 0 && boundaryCount < boundary.Length)
                    boundary[boundaryCount++] = new Point(right, y);
            }
            for (int direction = -1; direction <= 1; direction += 2)
            {
                int neighborY = y + direction;
                if ((uint)neighborY >= (uint)mapHeight) continue;
                int row = neighborY * mapWidth, x = left;
                while (x <= right)
                {
                    if (bitmap[row + x] != 0)
                    {
                        if (boundary is not null && boundaryCount < boundary.Length)
                            boundary[boundaryCount++] = new Point(x, y);
                        x++;
                    }
                    else if (visited[row + x] == 0)
                    {
                        int runLeft = x, runRight = x;
                        while (runLeft > 0 && bitmap[row + runLeft - 1] == 0 && visited[row + runLeft - 1] == 0) runLeft--;
                        while (runRight + 1 < mapWidth && bitmap[row + runRight + 1] == 0 && visited[row + runRight + 1] == 0) runRight++;
                        ArrayCompat.Fill(visited, (byte)1, row + runLeft, runRight - runLeft + 1);
                        stack[top++] = neighborY; stack[top++] = runLeft; stack[top++] = runRight;
                        x = runRight + 1;
                    }
                    else x++;
                }
            }
        }
    }

    private static bool TryBuildDetection(Point[] contour, int count, ReadOnlySpan<float> prediction,
        int mapWidth, int mapHeight, PaddleOcrDetectorOptions options, int sourceWidth, int sourceHeight,
        Point[] hullScratch, Span<Point> corners, Span<Point> miniBoxScratch,
        Span<Point> expandedMiniBoxScratch, Span<Point> remappedCornersScratch,
        out PaddleOcrDetectionBox detection)
    {
        detection = default;
        if (count < 3) return false;
        if (!TryMinimumRectangle(contour, count, hullScratch,
            out Rectangle rectangle, miniBoxScratch)) return false;
        float shortestSide = MathF.Min(rectangle.MaxU - rectangle.MinU, rectangle.MaxV - rectangle.MinV);
        if (shortestSide < 3) return false;
        float score = PolygonScore(prediction, mapWidth, mapHeight, miniBoxScratch);
        if (!MathCompat.IsFinite(score) || score < options.BoxThreshold) return false;
        Point[] unclipped = Unclip(miniBoxScratch, options.UnclipRatio);
        if (unclipped.Length < 3) return false;
        if (!TryMinimumRectangle(unclipped, unclipped.Length, hullScratch,
            out rectangle, expandedMiniBoxScratch)) return false;
        shortestSide = MathF.Min(rectangle.MaxU - rectangle.MinU, rectangle.MaxV - rectangle.MinV);
        if (shortestSide < 5) return false;
        RectanglePoints(rectangle, 0, corners); OrderClockwise(corners);
        for (int i = 0; i < 4; i++)
        {
            // PaddleOCR's quad post-process maps coordinates back to the
            // source image with np.round, then clips to [0, size].
            float x = MathF.Round(corners[i].X / mapWidth * sourceWidth,
                MidpointRounding.ToEven);
            float y = MathF.Round(corners[i].Y / mapHeight * sourceHeight,
                MidpointRounding.ToEven);
            corners[i] = new Point(Clamp(x, sourceWidth), Clamp(y, sourceHeight));
        }
        // PaddleX's CropByPolys converts mapped polygons to int32 and calls
        // minAreaRect before perspective cropping. Recompute that rectangle
        // here so the pure-managed crop receives the same geometry.
        Point[] mappedPoints = corners.ToArray();
        if (!TryMinimumRectangle(mappedPoints, mappedPoints.Length, hullScratch,
            out _, remappedCornersScratch)) return false;
        remappedCornersScratch.CopyTo(corners);
        if (Distance(corners[0], corners[1]) <= 4 || Distance(corners[0], corners[3]) <= 4)
            return false;
        detection = new PaddleOcrDetectionBox(corners[0].X, corners[0].Y, corners[1].X, corners[1].Y,
            corners[2].X, corners[2].Y, corners[3].X, corners[3].Y, score);
        return true;
    }

    private static void Dilate2x2(byte[] bitmap, byte[] dilated, int width, int height, int pixels)
    {
        Array.Clear(dilated, 0, pixels);
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
        {
            int dst = y * width + x;
            // OpenCV's default anchor for a 2x2 kernel is (0, 0), so the
            // source footprint for dst(x,y) is (x,y), (x+1,y),
            // (x,y+1), (x+1,y+1). This is the same as cv2.dilate(...,
            // np.ones((2, 2), np.uint8)).
            if (bitmap[dst] != 0 || (x + 1 < width && bitmap[dst + 1] != 0) ||
                (y + 1 < height && bitmap[dst + width] != 0) ||
                (x + 1 < width && y + 1 < height && bitmap[dst + width + 1] != 0))
                dilated[dst] = 1;
        }
        Buffer.BlockCopy(dilated, 0, bitmap, 0, pixels);
    }

    private static bool IsBoundary(byte[] bitmap, int width, int height, int x, int y)
    {
        if (x == 0 || y == 0 || x + 1 == width || y + 1 == height) return true;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
            if ((dx != 0 || dy != 0) && bitmap[(y + dy) * width + x + dx] == 0) return true;
        return false;
    }

    private static Point[] Unclip(ReadOnlySpan<Point> box, float ratio)
    {
        int n = box.Length;
        if (n < 3) return [];
        double area = Math.Abs(SignedArea(box)), perimeter = 0;
        for (int i = 0; i < n; i++) perimeter += Distance(box[i], box[(i + 1) % n]);
        double distance = area * ratio / perimeter;
        if (!MathCompat.IsFinite(distance) || distance <= 0) return [];
        bool profile = PipelineProfiler.Enabled;
        long started = profile ? PipelineProfiler.Now() : 0;
        Point[] expanded = OffsetRound(box, distance);
        if (profile) PipelineProfiler.Add(PipelineProfiler.DetUnclip, started);
        return expanded.Length >= 3 ? expanded : [];
    }

    /// <summary>
    /// Convex outward offset with round joins. Same construction as Clipper 5.1.5
    /// <c>PolyOffsetBuilder</c> for a single positive-delta polygon, without the
    /// trailing boolean union (unnecessary once the caller takes a min-area rect).
    /// </summary>
    private static Point[] OffsetRound(ReadOnlySpan<Point> box, double delta)
    {
        int n = box.Length;
        Span<(long X, long Y)> pts = stackalloc (long, long)[n];
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            long x = (long)Math.Round(box[i].X, MidpointRounding.ToEven);
            long y = (long)Math.Round(box[i].Y, MidpointRounding.ToEven);
            if (count > 0 && pts[count - 1].X == x && pts[count - 1].Y == y) continue;
            pts[count++] = (x, y);
        }
        if (count >= 2 && pts[0].X == pts[count - 1].X && pts[0].Y == pts[count - 1].Y) count--;
        if (count < 3) return [];

        double winding = 0;
        for (int i = 0; i < count; i++)
        {
            (long X, long Y) = pts[i];
            (long X, long Y) b = pts[(i + 1) % count];
            winding += (double)X * b.Y - (double)b.X * Y;
        }
        if (winding < 0) pts[..count].Reverse();

        Span<(double X, double Y)> normals = stackalloc (double, double)[count];
        for (int i = 0; i < count; i++)
        {
            (long X, long Y) = pts[i];
            (long X, long Y) b = pts[(i + 1) % count];
            double dx = b.X - X, dy = b.Y - Y;
            double len = Math.Sqrt(dx * dx + dy * dy);
            if (len > 0) { dx /= len; dy /= len; }
            normals[i] = (dy, -dx);
        }

        List<Point> result = [with(count * 6)];
        int prev = count - 1;
        for (int i = 0; i < count; i++)
        {
            (long X, long Y) = pts[i];
            (double X, double Y) incoming = normals[prev];
            (double X, double Y) outgoing = normals[i];
            AddRounded(result, X + incoming.X * delta, Y + incoming.Y * delta);
            if ((incoming.X * outgoing.Y - outgoing.X * incoming.Y) * delta >= 0)
            {
                if (outgoing.X * incoming.X + outgoing.Y * incoming.Y < 0.985)
                {
                    double start = Math.Atan2(incoming.Y, incoming.X);
                    double end = Math.Atan2(outgoing.Y, outgoing.X);
                    if (end < start) end += Math.PI * 2;
                    AddArc(result, X, Y, start, end, delta);
                }
            }
            else AddRounded(result, X, Y);
            AddRounded(result, X + outgoing.X * delta, Y + outgoing.Y * delta);
            prev = i;
        }
        return [.. result];
    }

    private static void AddArc(List<Point> dst, double cx, double cy, double start, double end, double radius)
    {
        double radiusAbs = Math.Abs(radius);
        double chord = Math.Min(0.25, radiusAbs);
        double fraction = Math.Abs(end - start) / (2 * Math.PI);
        if (fraction <= 0 || radiusAbs <= 0) return;
        int steps = (int)(fraction * Math.PI / Math.Acos(1 - chord / radiusAbs));
        if (steps < 2) steps = 2;
        int cap = (int)(222.0 * fraction);
        if (steps > cap) steps = Math.Max(2, cap);
        double x = Math.Cos(start), y = Math.Sin(start);
        double step = (end - start) / steps;
        double cos = Math.Cos(step), sin = Math.Sin(step);
        for (int i = 0; i <= steps; i++)
        {
            AddRounded(dst, cx + x * radius, cy + y * radius);
            double nx = x * cos - y * sin;
            y = x * sin + y * cos;
            x = nx;
        }
    }

    private static void AddRounded(List<Point> dst, double x, double y) =>
        dst.Add(new Point(RoundHalfAway(x), RoundHalfAway(y)));

    private static float RoundHalfAway(double value) =>
        value < 0 ? (long)(value - 0.5) : (long)(value + 0.5);

    private static double SignedArea(ReadOnlySpan<Point> points)
    {
        double area = 0; for (int i = 0; i < points.Length; i++)
        { Point a = points[i], b = points[(i + 1) % points.Length]; area += (double)a.X * b.Y - (double)b.X * a.Y; }
        return area * 0.5;
    }

    private static bool TryMinimumRectangle(Point[] points, int count, Point[] hullScratch,
        out Rectangle rectangle, Span<Point> corners)
    {
        rectangle = default; int hullCount = ConvexHull(points, count, hullScratch);
        if (hullCount < 3) return false;
        float bestArea = float.PositiveInfinity;
        for (int edge = 0; edge < hullCount; edge++)
        {
            Point current = hullScratch[edge], next = hullScratch[(edge + 1) % hullCount];
            float dx = next.X - current.X, dy = next.Y - current.Y, length = MathF.Sqrt(dx * dx + dy * dy);
            if (length <= 0) continue;
            float ux = dx / length, uy = dy / length, vx = -uy, vy = ux;
            float minU = float.PositiveInfinity, maxU = float.NegativeInfinity, minV = float.PositiveInfinity, maxV = float.NegativeInfinity;
            for (int i = 0; i < hullCount; i++)
            { float pu = hullScratch[i].X * ux + hullScratch[i].Y * uy, pv = hullScratch[i].X * vx + hullScratch[i].Y * vy; minU = MathF.Min(minU, pu); maxU = MathF.Max(maxU, pu); minV = MathF.Min(minV, pv); maxV = MathF.Max(maxV, pv); }
            float area = (maxU - minU) * (maxV - minV);
            if (area < bestArea) { bestArea = area; rectangle = new Rectangle { Ux = ux, Uy = uy, Vx = vx, Vy = vy, MinU = minU, MaxU = maxU, MinV = minV, MaxV = maxV }; }
        }
        if (!MathCompat.IsFinite(bestArea) || bestArea <= 0) return false;
        if (corners.Length < 4) throw new ArgumentException("Corner buffer is too small.", nameof(corners));
        RectanglePoints(rectangle, 0, corners); OrderClockwise(corners); return true;
    }

    private static int ConvexHull(Point[] points, int pointCount, Point[] hull)
    {
        Array.Sort(points, 0, pointCount, Comparer<Point>.Create(static (a, b) => a.X < b.X ? -1 : a.X > b.X ? 1 : a.Y < b.Y ? -1 : a.Y > b.Y ? 1 : 0));
        int unique = 0; for (int i = 0; i < pointCount; i++) if (unique == 0 || points[i] != points[unique - 1]) points[unique++] = points[i];
        if (unique < 3) return 0;
        int count = 0; for (int i = 0; i < unique; i++) { while (count >= 2 && Cross(hull[count - 2], hull[count - 1], points[i]) <= 0) count--; hull[count++] = points[i]; }
        int lowerCount = count; for (int i = unique - 1; i > 0; i--) { while (count > lowerCount && Cross(hull[count - 2], hull[count - 1], points[i - 1]) <= 0) count--; hull[count++] = points[i - 1]; }
        return count > 1 ? count - 1 : 0;
    }

    private static float Cross(Point origin, Point first, Point second) => (first.X - origin.X) * (second.Y - origin.Y) - (first.Y - origin.Y) * (second.X - origin.X);
    private static Point FromProjection(Rectangle r, float u, float v) => new(u * r.Ux + v * r.Vx, u * r.Uy + v * r.Vy);
    private static void RectanglePoints(Rectangle r, float expansion, Span<Point> p) { p[0] = FromProjection(r, r.MinU - expansion, r.MinV - expansion); p[1] = FromProjection(r, r.MaxU + expansion, r.MinV - expansion); p[2] = FromProjection(r, r.MaxU + expansion, r.MaxV + expansion); p[3] = FromProjection(r, r.MinU - expansion, r.MaxV + expansion); }
    private static void OrderClockwise(Span<Point> p)
    {
        // Same sum/difference ordering as TextDetector.order_points_clockwise
        // in PaddleOCR (top-left, top-right, bottom-right, bottom-left).
        Span<Point> input = stackalloc Point[4];
        p.CopyTo(input);
        int minSumIndex = 0, maxSumIndex = 0;
        float minSum = input[0].X + input[0].Y, maxSum = minSum;
        for (int i = 1; i < 4; i++)
        {
            float sum = input[i].X + input[i].Y;
            if (sum < minSum) { minSum = sum; minSumIndex = i; }
            if (sum > maxSum) { maxSum = sum; maxSumIndex = i; }
        }
        Point topLeft = input[minSumIndex], bottomRight = input[maxSumIndex];
        int firstRemaining = -1, secondRemaining = -1;
        for (int i = 0; i < 4; i++)
        {
            if (i == minSumIndex || i == maxSumIndex) continue;
            if (firstRemaining < 0) firstRemaining = i;
            else secondRemaining = i;
        }
        Point first = input[firstRemaining], second = input[secondRemaining];
        float firstDiff = first.Y - first.X, secondDiff = second.Y - second.X;
        Point topRight = firstDiff < secondDiff ? first : second;
        Point bottomLeft = firstDiff < secondDiff ? second : first;
        p[0] = topLeft; p[1] = topRight; p[2] = bottomRight; p[3] = bottomLeft;
        if (!IsPositiveConvexCycle(p)) OrderByCentroidAngle(p, input);
    }

    // For a rectangle, x+y is minimal and maximal at opposite corners unless an
    // edge runs at exactly 45 degrees, where both corner pairs tie.  argmin and
    // argmax then land on *adjacent* corners and the order above traverses a
    // self-intersecting "bow-tie".  Such a quad has no projective map onto a
    // rectangle: PPOCRCrop's denominator g*u + h*v + 1 reaches zero inside the
    // crop grid and the warp throws.  Both OrderClockwise call sites receive an
    // exact minAreaRect, so this only fires on that 45-degree tie and official
    // parity is preserved for every other box.
    private static bool IsPositiveConvexCycle(ReadOnlySpan<Point> p)
    {
        // Image coordinates are y-down, so top-left -> top-right -> bottom-right
        // -> bottom-left turns in the positive direction at every corner.
        for (int i = 0; i < 4; i++)
            if (Cross(p[i], p[(i + 1) & 3], p[(i + 2) & 3]) <= 0) return false;
        return true;
    }

    private static void OrderByCentroidAngle(Span<Point> p, ReadOnlySpan<Point> input)
    {
        float centerX = 0, centerY = 0;
        for (int i = 0; i < 4; i++) { centerX += input[i].X; centerY += input[i].Y; }
        centerX /= 4; centerY /= 4;
        Span<float> angles = stackalloc float[4];
        Span<int> order = stackalloc int[4];
        for (int i = 0; i < 4; i++)
        {
            angles[i] = MathF.Atan2(input[i].Y - centerY, input[i].X - centerX);
            order[i] = i;
        }
        // Ascending angle about the centroid walks the corners in that same
        // positive direction, so the cycle can no longer self-intersect.
        for (int i = 1; i < 4; i++)
        {
            int j = i - 1, index = order[i];
            while (j >= 0 && angles[order[j]] > angles[index]) { order[j + 1] = order[j]; j--; }
            order[j + 1] = index;
        }
        // Rotate the cycle to start at the top-left corner. The 45-degree tie is
        // broken toward the topmost corner so one crop orientation is picked
        // deterministically instead of depending on hull iteration order.
        int start = 0;
        for (int i = 1; i < 4; i++)
        {
            Point candidate = input[order[i]], best = input[order[start]];
            float candidateSum = candidate.X + candidate.Y, bestSum = best.X + best.Y;
            if (candidateSum < bestSum || (candidateSum == bestSum && (candidate.Y < best.Y ||
                (candidate.Y == best.Y && candidate.X < best.X))))
                start = i;
        }
        for (int i = 0; i < 4; i++) p[i] = input[order[(start + i) & 3]];
    }
    private static float PolygonScore(ReadOnlySpan<float> prediction, int width, int height, ReadOnlySpan<Point> polygon)
    {
        float minX = float.PositiveInfinity, maxX = float.NegativeInfinity,
            minY = float.PositiveInfinity, maxY = float.NegativeInfinity;
        for (int i = 0; i < polygon.Length; i++)
        {
            Point p = polygon[i];
            minX = MathF.Min(minX, p.X); maxX = MathF.Max(maxX, p.X);
            minY = MathF.Min(minY, p.Y); maxY = MathF.Max(maxY, p.Y);
        }
        int left = Math.Max(0, (int)MathF.Floor(minX)), right = Math.Min(width - 1, (int)MathF.Ceiling(maxX)), top = Math.Max(0, (int)MathF.Floor(minY)), bottom = Math.Min(height - 1, (int)MathF.Ceiling(maxY));
        if (left > right || top > bottom) return 0;
        // cv2.fillPoly receives an int32 polygon. NumPy's astype(int32)
        // truncates toward zero, so apply the same conversion before testing
        // pixels (the points are shifted by the clipped bounding-box origin).
        int n = polygon.Length;
        Span<int> polyX = stackalloc int[n], polyY = stackalloc int[n];
        for (int i = 0; i < n; i++)
        {
            polyX[i] = (int)polygon[i].X - left;
            polyY[i] = (int)polygon[i].Y - top;
        }
        // All raster coordinates are integers, so the previous per-pixel
        // PointInPolygon test decomposes exactly into, per row: (a) crossing
        // parity against float thresholds computed with the identical
        // expression, and (b) "on-edge" hits where the integer cross product
        // is exactly zero (|cross| < 1e-5 over integers means cross == 0).
        Span<float> thresholds = stackalloc float[n];
        Span<int> hitLow = stackalloc int[n], hitHigh = stackalloc int[n];
        double sum = 0; long count = 0;
        for (int y = top; y <= bottom; y++)
        {
            int py = y - top, thresholdCount = 0, hitCount = 0;
            for (int i = 0, j = n - 1; i < n; j = i++)
            {
                int ax = polyX[i], ay = polyY[i], bx = polyX[j], by = polyY[j];
                int dx = bx - ax, dy = by - ay;
                if ((ay > py) != (by > py))
                    thresholds[thresholdCount++] = (float)dx * (py - ay) / dy + ax;
                if (py < Math.Min(ay, by) || py > Math.Max(ay, by)) continue;
                if (dy == 0)
                {
                    // cross = -(py-ay)*dx is zero on this row (py == ay here).
                    hitLow[hitCount] = Math.Min(ax, bx); hitHigh[hitCount++] = Math.Max(ax, bx);
                }
                else
                {
                    long numerator = (long)(py - ay) * dx;
                    if (numerator % dy == 0)
                    {
                        int hx = ax + (int)(numerator / dy);
                        if (hx >= Math.Min(ax, bx) && hx <= Math.Max(ax, bx))
                        { hitLow[hitCount] = hx; hitHigh[hitCount++] = hx; }
                    }
                }
            }
            for (int i = 1; i < thresholdCount; i++)
            {
                float value = thresholds[i]; int j = i - 1;
                while (j >= 0 && thresholds[j] > value) { thresholds[j + 1] = thresholds[j]; j--; }
                thresholds[j + 1] = value;
            }
            int rowBase = y * width, thresholdIndex = 0;
            for (int x = left; x <= right; x++)
            {
                int px = x - left;
                while (thresholdIndex < thresholdCount && thresholds[thresholdIndex] <= px) thresholdIndex++;
                bool inside = ((thresholdCount - thresholdIndex) & 1) != 0;
                if (!inside)
                    for (int h = 0; h < hitCount; h++)
                        if (px >= hitLow[h] && px <= hitHigh[h]) { inside = true; break; }
                if (inside) { sum += prediction[rowBase + x]; count++; }
            }
        }
        return count == 0 ? 0 : (float)(sum / count);
    }
    private static void SortReadingOrder(List<PaddleOcrDetectionBox> found)
    {
        // Match PaddleOCR's sorted_boxes: sort by the top-left point first,
        // then perform the stable local insertion pass for lines on the same
        // row (within ten pixels).
        found.Sort(static (a, b) =>
        {
            int y = a.Y1.CompareTo(b.Y1);
            return y != 0 ? y : a.X1.CompareTo(b.X1);
        });
        for (int i = 0; i < found.Count - 1; i++)
        {
            for (int j = i; j >= 0; j--)
            {
                PaddleOcrDetectionBox current = found[j], next = found[j + 1];
                if (MathF.Abs(next.Y1 - current.Y1) >= 10 || next.X1 >= current.X1)
                    break;
                found[j] = next;
                found[j + 1] = current;
            }
        }
    }
    private static float Distance(Point a, Point b) => MathF.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
    private static float Clamp(float value, int limit) => value < 0 ? 0 : value > limit ? limit : value;
}
