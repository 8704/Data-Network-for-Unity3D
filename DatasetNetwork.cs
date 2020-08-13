using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Sirenix.OdinInspector;
using System.Linq;

[System.Serializable]
public class DatasetNetwork : Network {
    #region Inspector
    [Button(50)]
    [BoxGroup("Metrics/InOut/Calculate/FeedTestData")]
    public void FeedTestData() {
        (inputs, TRUTH) = dataset.GetData(Dataset.Dataclass.test);
        Train(inputs, TRUTH, true, false); 
    }
    #endregion
    Dataset dataset;
    [FoldoutGroup("Dataset Training")] public TextAsset data;
    [FoldoutGroup("Dataset Training")] public bool trainWithDataset;
    [FoldoutGroup("Dataset Training")] [Range(0.1f, 0.9f)] public float trainTestSplit = 0.7f;
    [FoldoutGroup("Dataset Training")] [Range(0.1f, 0.9f)] public float testCVSplit = 0.9f;
    [FoldoutGroup("Dataset Training")] public bool shuffle = true;
    [FoldoutGroup("Dataset Training")] public bool normalize = true;
    [FoldoutGroup("Dataset Training")] [EnableIf("normalize")] public bool normalizeFromNegativeOne = true;
    [FoldoutGroup("Dataset Training")] public Char splitBy = ',';
    [FoldoutGroup("Dataset Training")] public IndependentVariable[] xColumns;

    public void Start() {
        Initialize();
    }
    public void Update() {
        TrainData();
    }
    public override void Initialize() {
        if (trainWithDataset)
            InitializeData();
        base.Initialize();
    }
    #region Dataset
    public void InitializeData() {
        GameObject dataGameObject = new GameObject("Data Holder");
        dataGameObject.transform.parent = gameObject.transform;
        dataset = dataGameObject.AddComponent<Dataset>();
        dataset.Initialize(data, xColumns, splitBy, trainTestSplit, testCVSplit, shuffle, normalize, normalizeFromNegativeOne);
        int xColumnLength = dataset.GetData(Dataset.Dataclass.train).Item1.Length;
        int yColumnLength = dataset.GetData(Dataset.Dataclass.train).Item2.Length;
        architecture[0] = new LayerArchitecture(xColumnLength, architecture[0].units, architecture[0].activation);
        for (int a = 1; a < architecture.Count; a++)
            architecture[a] = new LayerArchitecture(0, architecture[a].units, architecture[a].activation);
        architecture[architecture.Count - 1] = new LayerArchitecture(0,
            yColumnLength,
            architecture[architecture.Count - 1].activation);
    }
    public void TrainData() {
        inputs = new float[0];
        if (UnityEngine.Random.Range(0, 100) < 1) {
            (inputs, TRUTH) = dataset.GetData(Dataset.Dataclass.test);
            Train(inputs, TRUTH, true, false);
            WriteToGraph(cost2.Average(), false);
        } else {
            (inputs, TRUTH) = dataset.GetData(Dataset.Dataclass.train);
            Train(inputs, TRUTH, true, true);
            if (updateTrainHistogram)
                WriteToGraph(cost2.Average(), true);
        }
    }
    #endregion
}
