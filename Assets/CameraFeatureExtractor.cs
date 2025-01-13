using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;    // Sentis namespace

public class CameraFeatureExtractor : MonoBehaviour
{
    public YoloV8NanoDetector yolov8NanoDetector;  // ↑ のクラス(実体)をインスペクタでアタッチ
    private WebCamTexture cameraTexture;

    public RawImage rawImage;
    public AspectRatioFitter fit;

    public GameObject boxContainer;
    public GameObject boxPrefab;

    int targetWidth = 640;
    int targetHeight = 640;

    void Start()
    {
        // カメラ初期化
        StartCoroutine(InitializeCamera());
    }

    IEnumerator InitializeCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.Log("No camera devices found");
            yield break;
        }

        // 適当なカメラを選択 (例: 背面カメラ)
        for (int i = 0; i < devices.Length; i++)
        {
            if (!devices[i].isFrontFacing)
            {
                cameraTexture = new WebCamTexture(devices[i].name, 640, 640);
                break;
            }
        }

        // 万が一背面カメラがない場合、最初のカメラ
        if (cameraTexture == null)
        {
            cameraTexture = new WebCamTexture(devices[0].name, 640, 640);
        }

        cameraTexture.Play();
        rawImage.texture = cameraTexture;

        // アスペクト比合わせ
        float ratio = (float)cameraTexture.width / (float)cameraTexture.height;
        fit.aspectRatio = ratio;

        yield return null;
    }

    void Update()
    {
        if (cameraTexture == null || !cameraTexture.didUpdateThisFrame)
            return;

        // カメラの回転や反転を修正
        float ratio = (float)cameraTexture.width / (float)cameraTexture.height;
        fit.aspectRatio = ratio;

        // 画像が上下反転していれば補正
        float scaleY = cameraTexture.videoVerticallyMirrored ? -1f : 1f;
        rawImage.rectTransform.localScale = new Vector3(1f, scaleY, 1f);

        // 回転
        int orient = -cameraTexture.videoRotationAngle;
        rawImage.rectTransform.localEulerAngles = new Vector3(0, 0, orient);

        RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight);
        Graphics.Blit(cameraTexture, rt);

        // rt を元に Texture2D を作成
        Texture2D resizedTex = new Texture2D(targetWidth, targetHeight, TextureFormat.RGBA32, false);
        RenderTexture.active = rt;
        resizedTex.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        resizedTex.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        // Color32[] を取得
        Color32[] pixels = resizedTex.GetPixels32();

        // YOLO 推論コルーチンを呼び出し (毎フレーム呼び出しは負荷が高い場合もあるので注意)
        StartCoroutine(yolov8NanoDetector.Detect(pixels, targetWidth, OnDetectComplete));
    }

    /// <summary>
    /// Yolov5Detector(Dummy for YOLOv8) からのコールバック
    /// 推論結果のバウンディングボックス一覧を受け取って、UI生成
    /// </summary>
    private void OnDetectComplete(List<BoundingBox> boxes)
    {
        // 既存の Box を削除
        foreach (Transform child in boxContainer.transform)
        {
            Destroy(child.gameObject);
        }

        // box を生成
        for (int i = 0; i < boxes.Count; i++)
        {
            var bb = boxes[i];
            Debug.Log($"Detected {bb.Label} ({bb.Confidence:0.00}) " +
                      $"at [{bb.Rect.x}, {bb.Rect.y}, {bb.Rect.width}, {bb.Rect.height}]");

            // 新たに UI オブジェクトを生成
            GameObject newBox = Instantiate(boxPrefab, boxContainer.transform);

            // 名前を割り当て (DEBUG 用)
            newBox.name = $"{bb.Label}_{bb.Confidence:0.00}";

            // ※ デフォルトアンカーや Pivot が (0.5, 0.5) の前提で位置を計算
            float xPos = bb.Rect.x - (yolov8NanoDetector.inputWidth / 2f);
            float yPos = bb.Rect.y - (yolov8NanoDetector.inputHeight / 2f);

            newBox.GetComponent<RectTransform>().localPosition = new Vector2(xPos, yPos);

            // サイズ調整 (適当に /100 など)
            float w = bb.Rect.width / 100f;
            float h = bb.Rect.height / 100f;

            newBox.GetComponent<RectTransform>().sizeDelta = new Vector2(w, h);

            // ラベル用テキストなどを付けるならここで
            // ...
        }
    }
}
