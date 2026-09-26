namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// The narrow session surface the OCR stages (DET/CLS/REC) consume. Both the
/// CPU interpreter (<see cref="InferenceSession"/>) and GPU-backed sessions
/// implement it, so the stages stay backend-agnostic.
/// </summary>
internal interface IOcrSession : IDisposable
{
    TensorShape InputShape { get; }
    TensorShape OutputShape { get; }
    /// <summary>Caller-visible input buffer; preprocess writes into it.</summary>
    Span<float> InputData { get; }
    ResizeWorkspace ResizeWorkspace { get; }
    bool InputIsNhwc { get; }
    int IntraOpThreads { get; set; }
    /// <summary>Largest input volume this session has ever been reshaped to.</summary>
    int HighWaterInputVolume { get; }
    /// <summary>Hint: only nodes before the CTC projection need to run.</summary>
    bool PlanForCtcProjection { get; set; }
    bool IsProfilingEnabled { get; }

    void Reshape(ReadOnlySpan<int> inputShape);
    ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input);
    /// <summary>
    /// Runs the graph up to the CTC vocab projection and exposes the operands
    /// for a fused MatMul+ArgMax. Returns false when the graph does not end in
    /// the expected Softmax/MatMul(+Add) shape.
    /// </summary>
    bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands);
    ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool outputIsLogits);
    void NoteProfile(OperatorId operation, long started, int nodeIndex);
}
