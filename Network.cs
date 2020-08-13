using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;
using Sirenix.OdinInspector;
using System.IO;
using UnityEditor;
using UnityEngine.UI;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using UnityEngine.UIElements;

[System.Serializable]
public class Network : MonoBehaviour {
    public enum ActivationType { ArcTan, Elu, Prelu, Relu, Sigmoid, Softplus, Tanh, Linear, Mix };
    public enum CostFunction { Hidden, MSE, Classification, Softmax };
    //------------------------------------------------------------------------------------------ (Layer)
    [System.Serializable]
    public class Layer {
        //------------------------------------------------------------------------------------------ (Node)
        [System.Serializable]
        public class Node {
            //------------------------------------------------------------------------------------------ (Node) Variables
            #region (Node) Variables
            [GUIColor(0.85f, 0.85f, 0.75f)] public float[] nubW;
            [FoldoutGroup("Debug")] public float[] nubWT;
            [FoldoutGroup("Debug")] public float[] nubDelta;
            [FoldoutGroup("Debug")] public float[] lastInputs;
            [FoldoutGroup("Debug")] public float output;
            [FoldoutGroup("Debug")] public ActivationType activationType;
            
            [HideInInspector] [FoldoutGroup("Debug")] public int forwardlink;
            [HideInInspector] [FoldoutGroup("Debug")] public int layerIndex;
            [HideInInspector] [FoldoutGroup("Debug")] public int nodeIndex;
            [NonSerialized] [HideInInspector] public Network network;
            [NonSerialized] [HideInInspector] public Layer layer;
            delegate float ActivationOutput(float input);
            ActivationOutput activationOutput;
            delegate float ActivationDerivation(float output);
            ActivationDerivation activationDerivation;
            #endregion
            //------------------------------------------------------------------------------------------ (Node) Initializations
            #region (Node) Initializations
            public Node(int nubesAmount, int inputNubesAmount, Layer layer_, Network network_, int nodeIndex_, int layerIndex_, ActivationType activationType_, Node[] BackLinks = null) {
                layer = layer_;
                network = network_;
                nodeIndex = nodeIndex_;
                layerIndex = layerIndex_;
                nubW = new float[1 + nubesAmount + inputNubesAmount];
                nubWT = new float[1 + nubesAmount + inputNubesAmount];
                nubDelta = new float[1 + nubesAmount + inputNubesAmount];
                activationType = activationType_;
                for (int a = 0; a < nubW.Length; a++) {
                    nubW[a] = network.ClaimWeight(nubW.Length);
                    if (a - inputNubesAmount >= 1)
                        BackLinks[a - inputNubesAmount - 1].forwardlink = a;
                }
                #region
                if (!layer.isFinalLayer) {
                    if(activationType == ActivationType.ArcTan) {
                        activationOutput = Stats.ArcTan;
                        activationDerivation = Stats.ArcTan_;
                    }
                    if (activationType == ActivationType.Elu) {
                        activationOutput = Stats.Elu;
                        activationDerivation = Stats.Elu_;
                    }
                    if (activationType == ActivationType.Linear) {
                        activationOutput = Stats.Linear;
                        activationDerivation = Stats.Linear_;
                    }
                    if (activationType == ActivationType.Prelu) {
                        activationOutput = Stats.Prelu;
                        activationDerivation = Stats.Prelu_;
                    }
                    if (activationType == ActivationType.Relu) {
                        activationOutput = Stats.Relu;
                        activationDerivation = Stats.Relu_;
                    }
                    if (activationType == ActivationType.Sigmoid) {
                        activationOutput = Stats.Sigmoid;
                        activationDerivation = Stats.Sigmoid_;
                    }
                    if (activationType == ActivationType.Softplus) {
                        activationOutput = Stats.Softplus;
                        activationDerivation = Stats.Softplus_;
                    }
                    if (activationType == ActivationType.Tanh) {
                        activationOutput = Stats.Tanh;
                        activationDerivation = Stats.Tanh_;
                    }
                }
                #endregion
            }
            #endregion
            //------------------------------------------------------------------------------------------ (Node) Calculate Output
            #region (Node) Calculate Output
            public void CalculateOutput() {
                if (!layer.isFinalLayer)
                    output = activationOutput(GatherNubsOutputs());
                else
                    output = network.CostOutput(GatherNubsOutputs());
            }
            public float GatherNubsOutputs() {
                lastInputs = layer.allInputs;
                float[] outputData = new float[nubW.Length];
                for (int a = 0; a < nubW.Length; a++) {
                    outputData[a] = lastInputs[a] * nubW[a];
                }
                return outputData.Sum();
            }
            #endregion
            //------------------------------------------------------------------------------------------ (Node) Backpropogate
            #region (Node) Backpropogate
            public float GatherDeltasOfForwardNubs() {
                float delta = 0;
                for (int a = 0; a < network.layers[layerIndex + 1].nodes.Length; a++) {
                    delta += network.layers[layerIndex + 1].nodes[a].nubDelta[forwardlink];
                }
                return delta;
            }
            public void Backpropogate() {
                float derivation = 0;
                if (!layer.isFinalLayer) {
                    derivation = activationDerivation(output) * GatherDeltasOfForwardNubs();
                } else
                    derivation = network.CostDerivation(output, network.TRUTH[nodeIndex]);

                for (int a = 0; a < nubW.Length; a++) {
                    nubDelta[a] = (derivation * nubW[a]);
                    nubWT[a] += (derivation * lastInputs[a]);
                    if (network.descent) {
                        float lambda = a == 0 ? 0 : network.adjustedLambda;
                        float newW = nubW[a] - network.adjustedLr * nubWT[a];// + lambda * nubW[a];
                        nubW[a] = Mathf.Clamp(newW, nubW[a] * (1f - network.epsilon), nubW[a] * (1f + network.epsilon));
                        nubW[a] = newW;
                        nubWT[a] = 0;
                    }
                }
                /*
                if (network.descent) {
                    for (int a = 0; a < nubW.Length; a++) {
                        nubDelta[a] = (derivation * nubW[a]);
                        nubW[a] -= (network.lr * (derivation * lastInputs[a]));
                    }
                }*/
            }
            #endregion
        }
        //------------------------------------------------------------------------------------------ (Layer) Variables
        #region Variables (Layer)
        public Node[] nodes;
        [HideInInspector] public int layerIndex;
        [HideInInspector] public float[] allInputs;
        [HideInInspector] public float[] userInputs;
        [HideInInspector] public int totalNodes;
        [HideInInspector] public bool isFinalLayer = false;
        [NonSerialized] [HideInInspector] public Network network;
        #endregion
        public Layer(int layerIndex_, int nodesAmount, int nubesAmount, int inputNubesAmount, Network network_, ActivationType activation = ActivationType.Linear, Node[] backLinks = null) {
            layerIndex = layerIndex_;
            totalNodes = nodesAmount;
            network = network_;
            nodes = new Node[totalNodes];
            userInputs = new float[inputNubesAmount];
            allInputs = new float[1 + inputNubesAmount + nubesAmount];
            for (int a = 0; a < totalNodes; a++) {
                nodes[a] = new Node(nubesAmount, inputNubesAmount, this, network_, a, layerIndex_, activation, backLinks);
            }
        }
        //------------------------------------------------------------------------------------------ (Layer) Calculate Output
        #region CalculateOutput (Layer)
        public void PrepareInputs() {
            allInputs[0] = 1f;
            for (int a = 0; a < userInputs.Length; a++)
                allInputs[a + 1] = userInputs[a];
            int b = 0;
            if (layerIndex != 0)
                for (int a = userInputs.Length; a < allInputs.Length - 1; a++) {
                    allInputs[a + 1] = network.layers[layerIndex - 1].nodes[b].output;
                    b++;
                }
        }
        public void CalculateOutput() {
            PrepareInputs();
            for (int a = 0; a < totalNodes; a++) {
                nodes[a].CalculateOutput();
            }
        }
        #endregion
        //------------------------------------------------------------------------------------------ (Layer) Backpropogate
        #region Backpropogate (Layer)
        public void Backpropogate() {
            if (isFinalLayer)
                if (totalNodes != network.TRUTH.Length)
                    Debug.Log("Truth to Output Mismatch.");

            for (int a = 0; a < totalNodes; a++)
                nodes[a].Backpropogate();
        }
        #endregion
    }
    //------------------------------------------------------------------------------------------ (Network)
    [System.Serializable]
    public struct LayerArchitecture {
        [HorizontalGroup("Group 1", LabelWidth = 80)] public int inputSize;
        [HorizontalGroup("Group 1", LabelWidth = 80)] public int units;
        [HorizontalGroup("Group 1", LabelWidth = 80)] public ActivationType activation;
        public LayerArchitecture(int inputSize_, int units_, ActivationType activation_) {
            inputSize = inputSize_;
            units = units_;
            activation = activation_;
        }
        public LayerArchitecture(string inputSize_, string units_, string activation_) {
            inputSize = int.Parse(inputSize_);
            units = int.Parse(units_);
            activation = (ActivationType)Enum.Parse(typeof(ActivationType), activation_);
        }
        public int InputSize {
            set { inputSize = value; }
        }
    }
    #region (Network) Inspector
    private void ResetSamplesCount() {
        totalSamples = 1;
    }
    private void ClearLayerComposition() {
        architecture = new List<LayerArchitecture>();
    }
    private void ClearLoadedWeights() {
        loadedWeights = new List<float>();
    }
    [GUIColor(0.65f, 0.65f, 0.45f)]
    [Button(50)]
    [BoxGroup("Configurations/Build")]
    public void TestBuild() {
        Initialize();
    }
    [GUIColor(0.65f, 0.65f, 0.55f)]
    [Button(50)]
    [HorizontalGroup("Metrics/InOut/Calculate")]
    [BoxGroup("Metrics/InOut/Calculate/Calculate")]
    public void Calculate() {
        Train(inputs, TRUTH, true, true);
    }
    #endregion
    //------------------------------------------------------------------------------------------ (Network) Variables
    #region (Network) Variables
    public List<Layer> layers;
    [FoldoutGroup("Metrics")]
    [HorizontalGroup("Metrics/Costs/Split")]
    [HorizontalGroup("Metrics/Costs", LabelWidth = 100)] public float[] cost;
    [HorizontalGroup("Metrics/Costs", LabelWidth = 100)] public float[] cost2;
    [HorizontalGroup("Metrics/TotalSamples", LabelWidth = 100)] [EnableIf("costFunction", CostFunction.Softmax)] public int choice;
    [HorizontalGroup("Metrics/TotalSamples", LabelWidth = 100)] [InlineButton("ResetSamplesCount", "Reset")] public float totalSamples; int samples;
    [BoxGroup("Metrics/InOut")] public float[] inputs;
    [HorizontalGroup("Metrics/InOut/Split")]
    [BoxGroup("Metrics/InOut/Split/Outputs")] public float[] outputs;
    [BoxGroup("Metrics/InOut/Split/Truth")] [ShowInInspector] public float[] TRUTH;

    [FoldoutGroup("Metrics/Histogram")] public int histogramPeriod = 10;
    [FoldoutGroup("Metrics/Histogram")] public int intervals = 10;
    [FoldoutGroup("Metrics/Histogram")] public bool useHistogram = true;
    [FoldoutGroup("Metrics/Histogram")] public Histogram histogram;
    [FoldoutGroup("Metrics/Histogram")] public float lineWidth = 1;
    [FoldoutGroup("Metrics/Histogram")] [Range(1, 100)] public int horizontalZoom = 8;
    [FoldoutGroup("Metrics/Histogram")] [Range(1, 20f)] public int verticalZoom = 4;
    [FoldoutGroup("Metrics/Histogram")] public float horizontalScroll = 0;
    [FoldoutGroup("Metrics/Histogram")] [Range(0f, 1f)] public float verticalScroll = 0;

    [FoldoutGroup("Configurations")] public CostFunction costFunction;
    [HorizontalGroup("Configurations/Setup", LabelWidth = 50)] public float lr = 0.01f;[HideInInspector] float adjustedLr;
    [HorizontalGroup("Configurations/Setup", LabelWidth = 50)] public float lambda = 0.00f;[HideInInspector] float adjustedLambda;
    [HorizontalGroup("Configurations/Setup", LabelWidth = 50)] public float epsilon = 0.2f;
    [HorizontalGroup("Configurations/Setup", LabelWidth = 50)] public int batch = 1;
    //[HorizontalGroup("Configurations/Setup3", LabelWidth = 100)] [MinMaxSlider(-5f, 5f)] public Vector2 weightInitializationRange = new Vector2(-1f, 1f);
    [BoxGroup("Configurations/SaveLoad")] [InlineButton("Save", "SAVE GRAPH")] public string graphName;
    [BoxGroup("Configurations/SaveLoad")] [InlineButton("Load", "LOAD GRAPH")] public TextAsset graph;
    [BoxGroup("Configurations/Build")] [InlineButton("ClearLayerComposition", "Clear")] public List<LayerArchitecture> architecture;
    [BoxGroup("Configurations/Build")] [InlineButton("ClearLoadedWeights", "Clear")] public List<float> loadedWeights = new List<float>();

    #region Optimization Variables
    protected bool updateTrainHistogram = false;
    Thread thread;
    [FoldoutGroup("Debug")] public int claimedNubs = 0;
    [FoldoutGroup("Debug")] public bool threadFinished = true;
    [FoldoutGroup("Debug")] public bool calculateOutput = true;
    [FoldoutGroup("Debug")] public bool backpropogate = true;
    [FoldoutGroup("Debug")] public bool descent = true;
    #endregion
    #endregion
    //------------------------------------------------------------------------------------------ (Network) Initialization
    #region (Network) Initialization
    public virtual void Initialize() {
        if (useHistogram)
            InitializeHistogram();
        claimedNubs = 0;
        layers = new List<Layer>();
        int maxNodeCount = 0;
        for (int a = 0; a < architecture.Count; a++) {
            layers.Add(new Layer(a,  //layer Index
                architecture[a].units, //nodes Amount
                a == 0 ? 0 : (architecture[a - 1].units), //nubesAmount  
                architecture[a].inputSize,
                this,
                architecture[a].activation,
                a == 0 ? null : layers[a - 1].nodes));
            maxNodeCount = Mathf.Max(1 + architecture[a].inputSize + architecture[a].units, maxNodeCount);
        }
        layers.Last().isFinalLayer = true;
        outputs = new float[layers[layers.Count - 1].nodes.Length];
        cost = new float[outputs.Length];
        cost2 = new float[outputs.Length];
    }
    public void InitializeHistogram() {
        GameObject trainhistogramObject = new GameObject(gameObject.name + "'s Train Histogram");
        histogram = trainhistogramObject.AddComponent<Histogram>();
        histogram.LinkToNetwork(this, true);
        histogram.LinkToNetwork(this, false);
    }
    public float ClaimWeight(int nubLength) {
        float returnWeight = 0;
        if(claimedNubs < loadedWeights.Count) {
            returnWeight = loadedWeights[claimedNubs];
        } else {
            returnWeight = Mathf.Clamp((Stats.RandomGaussian() / 3f) * (20f / nubLength), -1, 1);
        }
        /*
        float returnWeight = claimedNubs < loadedWeights.Count
                                            ? loadedWeights[claimedNubs]
                                            : UnityEngine.Random.Range(weightInitializationRange.x,
                                                                weightInitializationRange.y);*/
        claimedNubs++;
        return returnWeight;
    }
    #endregion
    //------------------------------------------------------------------------------------------ (Network) Training and Using
    #region (Network) Training and Using
    public void FeedInput(int layer, float[] IN) {
        if (IN.Length == architecture[layer].inputSize) {
            layers[layer].userInputs = IN;
        } else {
            Debug.Log("Input size does not match: " + IN.Length + ". Skipping.");
        }
    }
    public virtual void Train(float[] IN, float[] truth, bool calculateOutput_ = true, bool backpropogate_ = true){
        if (threadFinished) {
            threadFinished = false;
            inputs = IN;
            FeedInput(0, inputs);
            TRUTH = truth;
            calculateOutput = calculateOutput_; backpropogate = backpropogate_;
            thread = new Thread(new ThreadStart(ThreadTrain));
            thread.Start();
        }
    }
    public void ThreadTrain() {
        if (calculateOutput)
            CalculateOutput();
        if (backpropogate) {
            samples++;
            if (samples == batch) {
                TurnOnDescent();
            }
            Backpropogate();
            if(descent == true) {
                TurnOffDescent();
            }
        }
        RecordCost();
        threadFinished = true;
    }
    public void TurnOnDescent() {
        descent = true;
        adjustedLr = lr / batch;
        adjustedLambda = lambda / batch;
    }
    public void TurnOffDescent() {
        samples = 0;
        descent = false;
    }
    public float[] CalculateOutput() {
        for (int a = 0; a < layers.Count; a++)
            layers[a].CalculateOutput();
        if (costFunction == CostFunction.Softmax)
            NormalizeOutput();
        ToOutputs();
        return outputs;
    }
    protected void Backpropogate() {
        for (int a = layers.Count - 1; a >= 0; a--)
            layers[a].Backpropogate();
    }
    public virtual void RecordCost() {
        totalSamples++;
        switch (costFunction) {
            case (CostFunction.MSE):
                for (int a = 0; a < outputs.Length; a++) {
                    cost[a] = Stats.Squared(outputs[a] - TRUTH[a]);
                    cost2[a] = cost[a];
                }
                break;
            case (CostFunction.Classification):
                for (int a = 0; a < outputs.Length; a++) {
                    float t = TRUTH[a] > 0.5f ? 1 : 0;
                    cost[a] = t * Mathf.Log(outputs[a]) + (1f - t) * Mathf.Log(1 - outputs[a]);
                    cost2[a] = (TRUTH[a] > 0.5f ? 1 : 0) == (outputs[a] > 0.5f ? 1 : 0) ? 1 : 0;
                }
                break;
        }
    }
    public void WriteToGraph(float recordCost, bool isTrain = true) {
        if (!float.IsNaN(recordCost)) {
            if (isTrain) {
                histogram.UpdateGraph(recordCost, 0);
                updateTrainHistogram = false;
            } else {
                histogram.UpdateGraph(recordCost, 1);
                updateTrainHistogram = true;
            }
        }
    }
    public void ToOutputs() {
        for (int a = 0; a < layers[layers.Count - 1].totalNodes; a++)
            outputs[a] = layers[layers.Count - 1].nodes[a].output;
    }
    public virtual void NormalizeOutput() {
        float outputSum = 0;
        for (int a = 0; a < layers[layers.Count - 1].nodes.Length; a++) {
            outputSum += layers[layers.Count - 1].nodes[a].output;
        }
        for (int a = 0; a < layers[layers.Count - 1].nodes.Length; a++) {
            layers[layers.Count - 1].nodes[a].output /= outputSum;
        }
    }
    public virtual float CostOutput(float input) {
        float output = 0;
        switch (costFunction) {
            case CostFunction.MSE:
                output = input;
                break;
            case CostFunction.Classification:
                output = Mathf.Clamp(1f / (1f + (Mathf.Exp(-input))), 0.000001f, 0.999999f);
                break;
            case CostFunction.Softmax:
                output = Mathf.Exp(input);
                break;
        }
        return output;
    }
    public void SoftmaxOutput() {
        ToOutputs();
        float outputSum = 0;
        for (int a = 0; a < outputs.Length; a++) {
            layers[layers.Count - 1].nodes[a].output -= outputs.Max();
            layers[layers.Count - 1].nodes[a].output = Mathf.Exp(layers[layers.Count - 1].nodes[a].output);
            outputSum += layers[layers.Count - 1].nodes[a].output;
        }
        for (int a = 0; a < outputs.Length; a++) {
            layers[layers.Count - 1].nodes[a].output /= outputSum;
        }
    }
    public virtual float CostDerivation(float output, float truth) {
        float derivation = 0;
        switch (costFunction) {
            case CostFunction.MSE:
                derivation = (output - truth);
                break;
            case CostFunction.Classification:
                float t = truth > 0.5f ? 1 : 0;
                derivation = output - t;
                break;
            case CostFunction.Softmax:
                derivation = output * ((output == outputs[choice] ? 1f : 0f) - outputs[choice]) * truth;
                break;
        }
        return derivation;
    }
    #endregion
    //------------------------------------------------------------------------------------------ (Network) Save
    #region (Network) Save
    public void Save() {
        string serializedData = SaveNetworkConfiguration();
        serializedData += SaveLayersComposition();
        serializedData += SaveWeights();
        StreamWriter writer = new StreamWriter("Assets/NN/Saves/" + graphName + ".txt", false);
        writer.Write(serializedData);
        writer.Close();
    }
    public string SaveNetworkConfiguration() {
        return costFunction + "," + lr + "," + lambda + "," + epsilon + "," + batch + "," + "\n";
    }
    public string SaveLayersComposition() {
        string serializedData = "";
        for (int a = 0; a < architecture.Count; a++) {
            serializedData += architecture[a].inputSize + ",";
            serializedData += architecture[a].units + ",";
            serializedData += architecture[a].activation + ",";
        }
        serializedData += "\n";
        return serializedData;
    }
    public string SaveWeights() {
        string serializedData = "";
        for (int a = 0; a < layers.Count; a++) {
            for (int b = 0; b < layers[a].nodes.Length; b++) {
                for (int c = 0; c < layers[a].nodes[b].nubW.Length; c++) {
                    serializedData += layers[a].nodes[b].nubW[c] + "\n";
                }
            }
        }
        return serializedData;
    }
    #endregion
    //------------------------------------------------------------------------------------------ (Network) Load
    #region (Network) Load
    public void Load() {
        LoadNetworkConfigurations(graph.text);
        LoadLayersComposition(graph.text);
        LoadWeights(graph.text);
        Initialize();
    }
    public void LoadNetworkConfigurations(string loadedString) {
        string[] enterSplit = loadedString.Split('\n');
        string[] networkConfigString = enterSplit[0].Split(',');
        costFunction = (CostFunction)Enum.Parse(typeof(CostFunction), networkConfigString[0]);
        lr = float.Parse(networkConfigString[1]);
        lambda = float.Parse(networkConfigString[2]);
        epsilon = float.Parse(networkConfigString[3]);
        batch = int.Parse(networkConfigString[4]);
    }
    public void LoadLayersComposition(string loadedString) {
        architecture = new List<LayerArchitecture>();
        string[] enterSplit = loadedString.Split('\n');
        string[] layersString = enterSplit[1].Split(',');
        for (int a = 0; a < layersString.Length - 2; a += 3) {
            architecture.Add(new LayerArchitecture(layersString[a], layersString[a + 1], layersString[a + 2]));
        }
    }
    public void LoadWeights(string loadedString) {
        loadedWeights = new List<float>();
        string[] enterSplit = loadedString.Split('\n');
        for (int a = 2; a < enterSplit.Length - 1; a++) {
            string[] commaSplit = enterSplit[a].Split(',');
            loadedWeights.Add(float.Parse(commaSplit[0]));
        }
    }
    #endregion
}