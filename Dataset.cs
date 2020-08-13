using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;

public class Dataset : MonoBehaviour {
    List<List<float>> data;
    List<List<string>> stringdata;
    List<List<List<float>>> oneHotData;
    [SerializeField] (List<List<float>>, List<List<float>>) trainData;
    [SerializeField] (List<List<float>>, List<List<float>>) testData;
    [SerializeField] (List<List<float>>, List<List<float>>) cvData;
    public enum Dataclass { train, test, cv };
    public (float[], float[]) GetData(Dataclass dataclass = Dataclass.train, int index = -1) {
        (float[], float[]) returnData = (new float[0], new float[0]);
        switch (dataclass) {
            case Dataclass.train:
                if (index == -1)
                    index = UnityEngine.Random.Range(0, trainData.Item1.Count);
                returnData.Item1 = trainData.Item1[index].ToArray();
                returnData.Item2 = trainData.Item2[index].ToArray();
                break;
            case Dataclass.test:
                if (index == -1)
                    index = UnityEngine.Random.Range(0, testData.Item1.Count);
                returnData.Item1 = testData.Item1[index].ToArray();
                returnData.Item2 = testData.Item2[index].ToArray();
                break;
            case Dataclass.cv:
                if (index == -1)
                    index = UnityEngine.Random.Range(0, cvData.Item1.Count);
                returnData.Item1 = cvData.Item1[index].ToArray();
                returnData.Item2 = cvData.Item2[index].ToArray();
                break;
        }
        return returnData;
    }
    public void Initialize(TextAsset textdata_, IndependentVariable[] x_, Char splitBy, float trainTestSplit = 0.6f, float testCVSplit = 0.5f, bool shuffle = true, bool scale = true, bool fromNegativeOne = true) {
        stringdata = Unpack(textdata_.text, splitBy, x_);
        data = Unpack(Unpack(textdata_.text, splitBy, x_));
        oneHotData = OneHotUnpack(stringdata, x_);
        trainData = SplitXY(data, oneHotData, x_);
        if (shuffle)
            Shuffle(data);
        if (scale) {
            trainData.Item1 = Scale(trainData.Item1, fromNegativeOne);
            trainData.Item2 = Scale(trainData.Item2, fromNegativeOne);
        }
        (trainData, testData) = SplitTrainTest(trainData, trainTestSplit);
        (testData, cvData) = SplitTrainTest(testData, testCVSplit);
    }
    public List<List<String>> Unpack(string data, Char splitBy, IndependentVariable[] x_) {
        string[] data_ = data.Split('\n');
        List<List<String>> data__ = new List<List<String>>();
        for (int a = 0; a < data_.Length; a++) {
            data__.Add(new List<String>());
            string[] data_0 = data_[a].Split(splitBy);
            for (int b = 0; b < x_.Length; b++) {
                for(int c = 0; c < x_[b].trim.Length; c++) {
                    if (x_[b].trim[c] != string.Empty) {
                        if (data_0.Length > x_[b].column)
                            data_0[x_[b].column - 1] = data_0[x_[b].column - 1].Replace(x_[b].trim[c], "");
                    }
                }
            }
            data__[a] = data_0.ToList();
        }
        for (int a = 0; a < data__.Count; a++) {
            if (data__[a].Count != data__[0].Count) {
                data__.RemoveAt(a);
                Debug.Log("Removed row " + a + "because it does not match row 1's size. ");
            }
        }
        return data__;
    }
    public List<List<float>> Unpack(List<List<String>> data) {
        List<List<float>> data_ = new List<List<float>>();
        for (int a = 0; a < data.Count; a++) {
            data_.Add(new List<float>());
            for (int b = 0; b < data[a].Count; b++) {
                float temp;
                data_[a].Add(float.NaN);
                if (float.TryParse(data[a][b], out temp)) {
                    data_[a][b] = temp;
                }
            }
        }
        return data_;
    }
    public List<List<List<float>>> OneHotUnpack(List<List<String>> data, IndependentVariable[] x_) {
        List<List<List<float>>> returnOneHot = new List<List<List<float>>>();
        for (int a = 0; a < data.Count; a++) {
            returnOneHot.Add(new List<List<float>>());
            for (int b = 0; b < data[0].Count; b++) {
                returnOneHot[a].Add(new List<float>());
            }
        }
        foreach (IndependentVariable iV in x_) {
            if (iV.oneHot) {
                List<string> tempList = new List<string>();
                for (int a = 0; a < data.Count; a++) {
                    if (data[a].Count > iV.column - 1)
                        tempList.Add(data[a][iV.column - 1]);
                    else
                        tempList.Add("Null");
                }
                List<List<float>> oneHottedList = OneHotEncoder(tempList);
                for (int a = 0; a < oneHottedList.Count; a++) {
                    returnOneHot[a][iV.column - 1] = oneHottedList[a];
                }
            }
        }
        return returnOneHot;
    }
    public List<List<float>> OneHotEncoder(List<String> data) {
        List<List<float>> oneHotList = new List<List<float>>();
        var unique_items = new HashSet<string>(data);
        for (int a = 0; a < data.Count; a++) {
            oneHotList.Add(new List<float>());
        }
        foreach (string ui in unique_items) {
            Debug.Log(ui);
        }
        foreach (string ui in unique_items) {
            if (data.Where(s => s != null && s.StartsWith(ui)).Count() > (data.Count / 10)) {
                for (int a = 0; a < data.Count; a++) {
                    if (data[a] == ui)
                        oneHotList[a].Add(1);
                    else
                        oneHotList[a].Add(0);
                }
            }
        }
        return oneHotList;
    }
    public (List<List<float>>, List<List<float>>) SplitXY(List<List<float>> data, List<List<List<float>>> oneHotData, IndependentVariable[] x_) {
        List<List<float>> x = new List<List<float>>();
        List<List<float>> y = new List<List<float>>();
        int dataTossed = 0;

        for (int a = 0; a < data.Count; a++) {
            List<float> tempX = new List<float>();
            List<float> tempY = new List<float>();
            for (int b = 0; b < data[a].Count; b++) {
                for (int c = 0; c < x_.Length; c++) {
                    if (x_[c].column == b + 1) {
                        if (!x_[c].oneHot) {
                            if (x_[c].relativeTo != 0) {
                                if (!x_[c].isObjective)
                                    tempX.Add(data[a][b] / data[a][(int)x_[c].relativeTo - 1]);
                                else
                                    tempY.Add(data[a][b] / data[a][(int)x_[c].relativeTo - 1]);
                            } else {
                                if (!x_[c].isObjective)
                                    tempX.Add(data[a][b]);
                                else
                                    tempY.Add(data[a][b]);
                                if (x_[c].scalarColumns != null) {
                                    for (int d = 0; d < x_[c].scalarColumns.Length; d++)
                                        if (!x_[c].isObjective) {
                                            tempX.Add(data[a][b] *
                                                (x_[c].scalarColumns[d].x == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].x - 1]) *
                                                (x_[c].scalarColumns[d].y == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].y - 1]) *
                                                (x_[c].scalarColumns[d].z == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].z - 1]) *
                                                (x_[c].scalarColumns[d].w == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].w - 1]));
                                        } else {
                                            tempY.Add(data[a][b] *
                                                (x_[c].scalarColumns[d].x == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].x - 1]) *
                                                (x_[c].scalarColumns[d].y == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].y - 1]) *
                                                (x_[c].scalarColumns[d].z == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].z - 1]) *
                                                (x_[c].scalarColumns[d].w == 0 ? 1 : data[a][(int)x_[c].scalarColumns[d].w - 1]));
                                        }


                                }
                            }
                        } else {
                            if (!x_[c].isObjective)
                                tempX.AddRange(oneHotData[a][b]);
                            else
                                tempY.AddRange(oneHotData[a][b]);
                        }
                    }
                }
                //if (y_.Contains(b + 1))
                //    tempY.Add(data[a][b]);
            }
            //int xColumnLength = 0;
            //for (int b = 0; b < x_.Length; b++)
            //    xColumnLength += x_[b].GetInputLength();
            if (tempX.Contains(float.NaN) || tempY.Contains(float.NaN)) {
                dataTossed++;
            } else {
                x.Add(tempX);
                y.Add(tempY);
            }
        }
        Debug.Log(dataTossed + " unclean data excluded.");
        return (x, y);
    }
    public ((List<List<float>>, List<List<float>>), (List<List<float>>, List<List<float>>)) SplitTrainTest((List<List<float>>, List<List<float>>) data, float trainTestSplit) {
        (List<List<float>>, List<List<float>>) train = (new List<List<float>>(), new List<List<float>>());
        (List<List<float>>, List<List<float>>) test = (new List<List<float>>(), new List<List<float>>());
        float trainMax = data.Item1.Count * trainTestSplit;
        train.Item1.AddRange(data.Item1.GetRange(0, (int)trainMax));
        train.Item2.AddRange(data.Item2.GetRange(0, (int)trainMax));
        test.Item1.AddRange(data.Item1.GetRange((int)trainMax, data.Item1.Count - (int)trainMax - 1));
        test.Item2.AddRange(data.Item2.GetRange((int)trainMax, data.Item2.Count - (int)trainMax - 1));
        return (train, test);
    }
    public void Shuffle<T>(IList<T> list) {
        System.Random random = new System.Random();
        int n = list.Count;
        for (int i = list.Count - 1; i > 1; i--) {
            int rnd = random.Next(i + 1);
            T value = list[rnd];
            list[rnd] = list[i];
            list[i] = value;
        }
    }
    public List<List<float>> Scale(List<List<float>> data, bool fromNegativeOne = true) {
        float[] minValues = new float[data[0].Count];
        float[] maxValues = new float[data[0].Count];
        for (int a = 0; a < data.Count; a++) {
            for (int b = 0; b < data[a].Count; b++) {
                if (a == 0) {
                    minValues[b] = data[a][b];
                    maxValues[b] = data[a][b];
                } else {
                    minValues[b] = Mathf.Min(data[a][b], minValues[b]);
                    maxValues[b] = Mathf.Max(data[a][b], maxValues[b]);
                }
            }
        }
        for (int a = 0; a < data.Count; a++) {
            for (int b = 0; b < data[a].Count; b++) {
                data[a][b] = (data[a][b] - minValues[b]) / (maxValues[b] - minValues[b]);
                if (fromNegativeOne)
                    data[a][b] = (data[a][b] - 0.5f) * 2;
            }
        }
        return data;
    }
}

[System.Serializable]
public struct IndependentVariable {
    public int column;
    public Vector4[] scalarColumns;
    public int relativeTo;
    public bool oneHot;
    public bool isObjective;
    public string[] trim;
    //  public int GetInputLength() {
    //     return scalarColumns.Length + relativeColumns.Length + 1;
    //  }
}
