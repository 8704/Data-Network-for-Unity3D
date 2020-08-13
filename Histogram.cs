using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using Unity.Collections;
using UnityEngine;
using UnityEngine.UI;

public class Histogram : MonoBehaviour {
    /*[System.Serializable]
    public struct MovingAverage {
        public int intervals;
        public Gradient color;
        [HideInInspector] public LineRenderer lineRenderer;
        public MovingAverage(int intervals_, Gradient color_, float lineWidth_, Transform parent) {
            lineRenderer = new GameObject(parent.name + ". MA: " + intervals_).AddComponent<LineRenderer>();
            lineRenderer.transform.parent = parent;
            lineRenderer.material = Resources.Load<Material>("VertexColor");
            lineRenderer.colorGradient = color_;
            lineRenderer.useWorldSpace = false;
            lineRenderer.widthMultiplier = Mathf.Max(lineWidth_, 0.003f);
            lineRenderer.numCornerVertices = 5;
            lineRenderer.numCapVertices = 5;
            intervals = intervals_;
            color = color_;
        }
    }*/
    public List<List<float>> mydata = new List<List<float>>();
    public Color backgroundColor = Color.gray;
    [Range(1, 100)] public int horizontalZoom = 8;
    [Range(1, 20f)] public int verticalZoom = 4;
    public float horizontalScroll = 0;
    [Range(0f, 1f)] public float verticalScroll = 0;
    public List<LineRenderer> MAs = new List<LineRenderer>();
    public int intervals = 50;
    // public bool isTrainSet = true;
    [NonSerialized] Network network;
    public List<TextMeshPro> scoreText = new List<TextMeshPro>();
    public List<GameObject> textObject = new List<GameObject>();

    Camera cam;
    Plane plane;
    Plane frontPlane;
    public void UpdateGraph(float data, int index) {
        mydata[index].Add(data);
    }
    public void LinkToNetwork(Network network_, bool isTrainSet_) {
        network = network_;
        name = network.gameObject.name;

        MAs.Add(new GameObject(transform.name + ". MA").AddComponent<LineRenderer>());
        MAs[MAs.Count - 1].transform.parent = transform;
        MAs[MAs.Count - 1].material = Resources.Load<Material>("VertexColor");

        MAs[MAs.Count - 1].useWorldSpace = false;
        MAs[MAs.Count - 1].numCornerVertices = 5;
        MAs[MAs.Count - 1].numCapVertices = 5;
        GradientColorKey gck = new GradientColorKey();
        if (isTrainSet_)
            gck.color = Color.blue;
        else
            gck.color = Color.red;
        MAs[MAs.Count - 1].colorGradient.colorKeys[0] = gck;
        MAs[MAs.Count - 1].colorGradient.colorKeys[1] = gck;
        transform.parent = network.gameObject.transform;

        textObject.Add(new GameObject("TextObject"));
        scoreText.Add(textObject[textObject.Count - 1].AddComponent<TextMeshPro>());
        scoreText[scoreText.Count - 1].rectTransform.SetParent(this.transform, false);
        scoreText[scoreText.Count - 1].rectTransform.localScale = new Vector3(0.05f, 0.05f, 0);
        scoreText[scoreText.Count - 1].alignment = TextAlignmentOptions.MidlineRight;
        scoreText[scoreText.Count - 1].color = isTrainSet_ ? Color.blue : Color.red;
        mydata.Add(new List<float>());
    }

    #region Start
    public void Start() {
        SetUpCamera();
        SetUpPlane();
    }
    void SetUpCamera() {
        cam = new GameObject("Camera").AddComponent<Camera>();
        cam.transform.parent = this.transform;
        cam.transform.position += new Vector3(.9f, .5f, -10);
        cam.orthographic = true;
        cam.orthographicSize = .5f;
        cam.enabled = false;
    }
    void SetUpPlane() {
        GameObject newPlane = new GameObject("Back Plane");
        newPlane.transform.parent = this.transform;
        MeshFilter mf = newPlane.gameObject.AddComponent<MeshFilter>();
        newPlane.gameObject.AddComponent<MeshRenderer>();
        plane = new Plane();
        plane.Create(10, 1, 1, mf, 0, true);
        plane.AddColor(backgroundColor);

        GameObject newPlane2 = new GameObject("Front Plane");
        newPlane2.transform.parent = this.transform;
        MeshFilter mf2 = newPlane2.gameObject.AddComponent<MeshFilter>();
        newPlane2.gameObject.AddComponent<MeshRenderer>();
        frontPlane = new Plane();
        frontPlane.Create(10, 1, 1, mf2, -1, true);
        frontPlane.AddUniqueMaterial(Resources.Load<Material>("Graph"));
    }
    #endregion
    public void Update() {
        if (mydata[0].Count == 0) return;
        horizontalZoom = network.horizontalZoom;
        verticalZoom = network.verticalZoom;
        horizontalScroll = network.horizontalScroll;
        verticalScroll = network.verticalScroll;
        if (intervals != network.intervals) {
            intervals = network.intervals;
            RecalculateMovingAverages();
        }

        for (int a = 0; a < MAs.Count; a++) {
            MAs[a].transform.position = new Vector3(horizontalScroll, -verticalScroll, 1);
            MAs[a].transform.localScale = new Vector3(1f / (Mathf.Pow(2, horizontalZoom)), Mathf.Pow(2, verticalZoom), 1);
            MAs[a].widthMultiplier = (MAs[a].transform.localScale.x / (a == 1 ? 3 : 5)) * network.lineWidth;
            scoreText[a].transform.position = Vector3.Scale(MAs[a].GetPosition(MAs[a].positionCount - 1), MAs[a].transform.localScale);
            scoreText[a].transform.localScale = new Vector3(MAs[a].widthMultiplier, MAs[a].widthMultiplier, 1);
        }
        DataToGraph();
        if (mydata[0].Count > 50000)
            FreeUpData(10000);
    }
    public void RecalculateMovingAverages() {
        if (mydata[0].Count > intervals) {
            for (int a = 0; a < MAs.Count; a++) {
                MAs[a].positionCount = 0;
                for (int b = intervals; b < mydata[a].Count; b += intervals) {
                    MAs[a].positionCount++;
                    MAs[a].SetPosition(MAs[a].positionCount - 1, new Vector3(
                        b,
                        GetMovingAverage(intervals, a, b),
                        -1f));
                    scoreText[a].text = GetMovingAverage(intervals, a, b).ToString("0.0E0");
                }
            }
        }
    }
    public void DataToGraph() {
        for (int a = 0; a < mydata.Count; a++) {
            if (mydata[a].Count % intervals == 0) {
                if (MAs[a].GetPosition(MAs[a].positionCount - 1).x != mydata[a].Count)
                    MAs[a].positionCount++;
                MAs[a].SetPosition(MAs[a].positionCount - 1, new Vector3(
                    mydata[a].Count,
                    GetMovingAverage(intervals, a, mydata[a].Count),
                    -1f));
                scoreText[a].text = GetMovingAverage(intervals, a, mydata[a].Count).ToString("0.0E0");
            }
        }
    }
    public float GetMovingAverage(int amount, int index, int endDataIndex) {
        float MA = 0;
        amount = Mathf.Max(2, amount);
        for (int a = 0; a < amount; a++) {
            MA += mydata[index][endDataIndex - 1 - a];
        }
        MA /= amount;
        return MA;
    }
    public void FreeUpData(int amountToRemove = 1) {
        for (int a = 0; a < mydata.Count; a++) {
            mydata[a].RemoveRange(0, Mathf.Min(amountToRemove, mydata[a].Count));
        }
        RecalculateMovingAverages();
    }
}
