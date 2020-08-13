using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class Stats {
    public static int Add(int num1, int num2) {
        return num1 + num2;
    }
    public static float Squared(float number) {
        number *= number;
        return number;
    }
    public static float Sum(float[] Samples) {
        float sum = 0;
        foreach (float f in Samples) {
            sum += f;
        }
        return sum;
    }
    public static float Mean(float[] Samples) {
        float sum = Sum(Samples);
        float mean = sum / Samples.Length;
        return mean;
    }
    public static float Mean(float Sample, float Mean, int Count) {
        if (Count == 0) Count = 1;
        float difference = Sample - Mean;
        float mean = (difference / Count) + Mean;
        return mean;
    }
    public static float ErrorSquared(float Sample, float Truth) {
        float error = Sample - Truth;
        error *= error;
        return error;
    }
    public static float MSE(float[] Samples, float Truth) {
        float errorsum = 0;
        foreach (float f in Samples) {
            errorsum += ErrorSquared(f, Truth);
        }
        float mse = errorsum / Samples.Length;
        return mse;
    }
    public static float Variance(float[] Samples) {
        float Truth = Mean(Samples);
        float errorsum = 0;
        foreach (float f in Samples) {
            errorsum += ErrorSquared(f, Truth);
        }
        float mse = errorsum / Samples.Length;
        return mse;
    }
    public static float Variance_(float Sample, float Mean, float MSE, int Count) {
        float errorsquared = ErrorSquared(Sample, Mean);
        float variance = Stats.Mean(errorsquared, MSE, Count);
        return variance;
    }
    public static float STDV(float[] Samples) {
        float stdv = Mathf.Sqrt(Variance(Samples));
        return stdv;
    }
    public static float STDV_(float Sample, float Mean, float STDV, int Count) {
        float variance = Squared(STDV);
        float stdv = Mathf.Sqrt(Variance_(Sample, Mean, variance, Count));
        return stdv;
    }

    public static float StandardError(float[] Samples) {
        float standarderror = Stats.STDV(Samples) / Mathf.Sqrt(Samples.Length);
        return standarderror;
    }
    public static float StandardError_(float Sample, float Mean, float StandardError, int Count) {
        float standarderror = Stats.STDV_(Sample, Mean, StandardError, Count) / Mathf.Sqrt(Count);
        return standarderror;
    }
    public static Vector2 ConfidenceBound(float[] Samples) {
        float upperbound = Mean(Samples) + 1.96f * StandardError(Samples);
        float lowerbound = Mean(Samples) - 1.96f * StandardError(Samples);
        return (new Vector2(lowerbound, upperbound));
    }
    public static Vector2 ConfidenceBound(float[] Samples, float ConfidenceMultiplier) {
        float upperbound = Mean(Samples) + ConfidenceMultiplier * StandardError(Samples);
        float lowerbound = Mean(Samples) - ConfidenceMultiplier * StandardError(Samples);
        return (new Vector2(lowerbound, upperbound));
    }
    public static float SmoothStop(float Value, float Power) {
        float smoothstop = 1 - Mathf.Pow((1 - Value), Power);
        return smoothstop;
    }
    public static float SmoothStop(float Value) {
        float smoothstop = 1 - Mathf.Pow((1 - Value), 2.71828f);
        return smoothstop;
    }
    public static float SmoothStart(float Value, float Power) {
        float smoothstart = Mathf.Pow(Value, Power);
        return smoothstart;
    }
    public static float SmoothStart(float Value) {
        float smoothstart = Mathf.Pow(Value, 2.71828f);
        return smoothstart;
    }

    public static float RandomGaussian() {
        float u, v;
        float s; // this is the hypotenuse squared.
        do {
            u = Random.Range(-1f, 1f);
            v = Random.Range(-1f, 1f);
            s = (u * u) + (v * v);
        } while (!(s != 0 && s < 1)); // keep going until s is nonzero and less than one

        // TODO allow a user to specify how many random numbers they want!
        // choose between u and v for seed (z0 vs z1)
        float seed;
        if (Random.Range(0, 2) == 0) {
            seed = u;
        } else {
            seed = v;
        }
        // create normally distributed number.
        float z = seed * Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
        return z;
    }


    public static float Linear(float input) {
        return input;
    }
    public static float Linear_(float output) {
        return 1;
    }
    public static float Relu(float input) {
        return Mathf.Max(0.0f, input);
    }
    public static float Relu_(float output) {
        return output > 0 ? 1 : 0;
    }
    public static float Prelu(float input) {
        return input >= 0 ? input : 0.1f * input;
    }
    public static float Prelu_(float output) {
        return output >= 0 ? 1 : 0.1f;
    }
    public static float Elu(float input) {
        return input >= 0 ? input : 0.1f * (Mathf.Exp(input) - 1);
    }
    public static float Elu_(float output) {
        return output >= 0 ? 1 : output + 0.1f;
    }
    public static float Sigmoid(float input) {
        return 1f / (1f + Mathf.Exp(-input));
    }
    public static float Sigmoid_(float output) {
        return output * (1 - output);
    }
    public static float Tanh(float input) {
        return (2f / (1 + Mathf.Exp(-2f * input))) - 1f;
    }
    public static float Tanh_(float output) {
        return 1f - output * output;
    }
    public static float ArcTan(float input) {
        return 1f / Mathf.Tan(input);
    }
    public static float ArcTan_(float coldput) {
        return 1f / ((coldput * coldput) + 1);
    }
    public static float Softplus(float input) {
        return Mathf.Log(1 + Mathf.Exp(input));
    }
    public static float Softplus_(float coldput) {
        return 1f / (1f + Mathf.Exp(-coldput));
    }
    public static float GaussianLoss(float action, float mean, float variance) {
        return Mathf.Abs(action - mean) / (variance * variance);
        // return (((action - mean) * (action - mean)) / (2 * variance * variance)) - Mathf.Log(Mathf.Sqrt(2f*Mathf.PI*variance*variance));
        //return (((action - mean) * (action - mean))/ (variance * variance));
    }


}

