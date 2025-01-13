using UnityEngine;
using UnityEngine.UI;

public class FullScreenRawImage : MonoBehaviour
{
    void Start()
    {
        // RectTransformの設定
        RectTransform rectTransform = GetComponent<RectTransform>();
        if (rectTransform != null)
        {
            rectTransform.anchorMin = new Vector2(0, 0); // 下左
            rectTransform.anchorMax = new Vector2(1, 1); // 上右
            rectTransform.offsetMin = Vector2.zero; // 左下オフセット
            rectTransform.offsetMax = Vector2.zero; // 右上オフセット
        }
        else
        {
            Debug.LogError("RectTransform component not found.");
        }
    }
}
