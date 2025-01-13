using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;      // Sentis namespace
using UnityEngine.UI;

/// <summary>
/// YOLOv8 Nano 推論用クラス (Sentis 2.x対応版)
/// </summary>
public class YoloV8NanoDetector : MonoBehaviour
{
    [Header("Model / Label Info")]
    [SerializeField]
    private ModelAsset modelAsset;
    public TextAsset labelsAsset; // ラベル一覧(1行1ラベル)

    private Worker m_Worker;

    // 入力 Tensor
    private Tensor m_Input;

    // ラベルリスト
    private string[] labels;

    [Header("Input / Network Config")]
    public int inputWidth = 640;   // モデルが想定する入力幅
    public int inputHeight = 640;  // モデルが想定する入力高さ
    public int classCount = 40;    // クラス数

    [Header("Thresholds")]
    [Range(0f, 1f)]
    public float minConfidence = 0.25f;  // objectness と classConf の積がこの値以上で検出
    [Range(0f, 1f)]
    public float nmsIoU = 0.45f;        // NMS の IoU しきい値
    public int maxObjects = 20;         // NMS 後の最大検出数

    void OnEnable()
    {
        // ラベル読込
        if (labelsAsset != null)
        {
            labels = labelsAsset.text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        }

        // モデル読み込み
        var model = ModelLoader.Load(modelAsset);

        // Worker を生成 (2.x 系では WorkerFactory から作成する推奨 API)
        m_Worker = new Worker(model, BackendType.CPU);

        Debug.Log("[YoloV8NanoDetector] Model & Worker initialized (Sentis 2.x).");
    }

    void OnDisable()
    {
        // Worker / テンソルを解放
        if (m_Worker != null)
        {
            m_Worker.Dispose();
            m_Worker = null;
        }
        if (m_Input != null)
        {
            m_Input.Dispose();
            m_Input = null;
        }
    }

    /// <summary>
    /// WebCam 等から得た Color32[] を YOLOv8 Nano へ推論し、結果のバウンディングボックスを返す。
    /// (Schedule → WaitForCompletion → PeekOutput の流れ)
    /// </summary>
    public IEnumerator Detect(Color32[] color32Array, int originalWidth, Action<List<BoundingBox>> onComplete)
    {
        // 1) 入力用 Tensor を作成
        m_Input = TransformInput(color32Array, inputWidth, inputHeight, originalWidth);
        // 2) 推論の実行を Schedule
        //    - 複数入力があれば Dictionary<string, Tensor> で渡すことも可能
        m_Worker.Schedule(m_Input);

        // 3) 推論が完了するまで待つ
        //m_Worker.WaitForCompletion();

        // 4) 推論結果を取得
        //    - 複数出力がある場合は "output0" や "output1" の名前で指定してください
        var outputTensor = m_Worker.PeekOutput() as Tensor<float>;
        if (outputTensor == null)
        {
            Debug.LogError("No output found from YOLO model!");
            yield return null;
        }

        // GPU から CPU へデータをコピー
        var cpuTensor = outputTensor.ReadbackAndClone();

        // 5) バウンディングボックスを解析 (ParseYoloV8Output)
        var rawBoxes = ParseYoloV8Output(cpuTensor);
        var filtered = FilterBoundingBoxes(rawBoxes, maxObjects, nmsIoU);

        // 後始末
        cpuTensor.Dispose();
        m_Input?.Dispose();
        m_Input = null;

        // 6) コールバックに結果を返す
        onComplete?.Invoke(filtered);
        yield return null;
    }

    /// <summary>
    /// Color32[] → (1, height, width, 3) への変換 & 正規化
    /// </summary>
    Tensor TransformInput(Color32[] pic, int width, int height, int originalWidth)
    {
        float[] floatValues = new float[width * height * 3];
        Debug.Log(pic.Length);

        int beginning = (((pic.Length / originalWidth) - height) * originalWidth) / 2;
        int leftOffset = (originalWidth - width) / 2;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int index = beginning + leftOffset + j;
                var color = pic[index];

                int idxOut = (i * width + j) * 3;

                // [0,1] に正規化
                floatValues[idxOut + 0] = color.r / 255f;
                floatValues[idxOut + 1] = color.g / 255f;
                floatValues[idxOut + 2] = color.b / 255f;
            }
            beginning += originalWidth;
        }

        // 形状: (1, height, width, 3)
        return new Tensor<float>(new TensorShape(1, height, width, 3), floatValues);
    }

    /// <summary>
    /// YOLOv8 Nano の推論結果をパース
    /// shape = (1, N, 5+classCount) を想定して (cx,cy,w,h,objConf,classScores...) を取る
    /// </summary>
    List<BoundingBox> ParseYoloV8Output(Tensor<float> tensor)
    {
        var boxes = new List<BoundingBox>();

        var shape = tensor.shape;
        int numPredictions = shape[1];
        int numFeatures = shape[2];

        for (int i = 0; i < numPredictions; i++)
        {
            float x_center = tensor[0, i, 0];
            float y_center = tensor[0, i, 1];
            float w = tensor[0, i, 2];
            float h = tensor[0, i, 3];
            float objConf = tensor[0, i, 4];
            if (objConf < minConfidence) continue;

            // クラス確率
            int bestClassIdx = -1;
            float bestClassScore = 0f;
            for (int c = 5; c < numFeatures; c++)
            {
                float classConf = tensor[0, i, c];
                if (classConf > bestClassScore)
                {
                    bestClassScore = classConf;
                    bestClassIdx = c - 5;
                }
            }

            // finalScore = objConf * classConf
            float finalScore = objConf * bestClassScore;
            if (finalScore < minConfidence)
                continue;

            // ラベル名
            string label = (labels != null && bestClassIdx >= 0 && bestClassIdx < labels.Length)
                ? labels[bestClassIdx] : $"Class_{bestClassIdx}";

            // 左上に変換
            float x = x_center - w / 2f;
            float y = y_center - h / 2f;

            boxes.Add(new BoundingBox
            {
                Label = label,
                LabelIdx = bestClassIdx,
                Confidence = finalScore,
                Dimensions = new BoundingBoxDimensions
                {
                    X = x,
                    Y = y,
                    Width = w,
                    Height = h
                }
            });
        }
        return boxes;
    }

    /// <summary>
    /// NMS で重複削除
    /// </summary>
    List<BoundingBox> FilterBoundingBoxes(List<BoundingBox> boxes, int limit, float iouThreshold)
    {
        var isActive = new bool[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
            isActive[i] = true;

        boxes.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

        var results = new List<BoundingBox>();

        for (int i = 0; i < boxes.Count; i++)
        {
            if (!isActive[i])
                continue;

            var boxA = boxes[i];
            results.Add(boxA);

            if (results.Count >= limit)
                break;

            for (int j = i + 1; j < boxes.Count; j++)
            {
                if (!isActive[j])
                    continue;

                var boxB = boxes[j];
                float iou = ComputeIoU(boxA.Rect, boxB.Rect);
                if (iou > iouThreshold)
                {
                    isActive[j] = false;
                }
            }
        }

        return results;
    }

    /// <summary>
    /// IoU 計算
    /// </summary>
    float ComputeIoU(Rect a, Rect b)
    {
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        if (areaA <= 0 || areaB <= 0)
            return 0f;

        float interXmin = Mathf.Max(a.xMin, b.xMin);
        float interYmin = Mathf.Max(a.yMin, b.yMin);
        float interXmax = Mathf.Min(a.xMax, b.xMax);
        float interYmax = Mathf.Min(a.yMax, b.yMax);

        float interW = Mathf.Max(0, interXmax - interXmin);
        float interH = Mathf.Max(0, interYmax - interYmin);
        float interArea = interW * interH;
        return interArea / (areaA + areaB - interArea);
    }
}

/// <summary>
/// バウンディングボックス用の構造体
/// </summary>
[Serializable]
public class BoundingBoxDimensions
{
    public float X;
    public float Y;
    public float Width;
    public float Height;
}

[Serializable]
public class BoundingBox
{
    public string Label;
    public int LabelIdx;
    public float Confidence;
    public BoundingBoxDimensions Dimensions;

    public Rect Rect => new Rect(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height);
}
