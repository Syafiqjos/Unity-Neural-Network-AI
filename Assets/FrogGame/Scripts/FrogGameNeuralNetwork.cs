using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class FrogGameNeuralNetwork
{
    private static int InputCount = 2;

    private static Matrix input0;
    private static Matrix hidden1;
    private static Matrix output2;

    private static Matrix targetOutput;

    private static Matrix weight_0_1;
    private static Matrix weight_1_2;

    private static Matrix bias_1;
    private static Matrix bias_2;

    private static Matrix bias_weight_1;
    private static Matrix bias_weight_2;

    private static float totalError;
    private static float learning_rate = 0.5f;

    static FrogGameNeuralNetwork()
    {
        //Create Neural Network
        //Design
        //
        // [input] -> [hidden] -> [output]
        //

        //Unknown Value
        input0 = new Matrix(new float[2, 1]);
        hidden1 = new Matrix(new float[2, 1]);
        output2 = new Matrix(new float[2, 1]);

        targetOutput = new Matrix(new float[2, 1]);

        //Fixed Value
        bias_1 = new Matrix(new float[1, 1] { { 1.0f } } );
        bias_2 = new Matrix(new float[1, 1] { { 1.0f } } );

        //Random Value
        weight_0_1 = new Matrix(new float[2, 2], true);
        weight_1_2 = new Matrix(new float[2, 2], true);

        bias_weight_1 = new Matrix(new float[2, 1], true);
        bias_weight_2 = new Matrix(new float[2, 1], true);
    }

    public static int GetInputCount()
    {
        return InputCount;
    }

    public static Matrix BoolArrayToMatrix(bool[] input)
    {
        Matrix matrix = new Matrix(new float[input.Length, 1]);
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i])
            {
                matrix.SetCell(i, 0, 1.0f);
            } else
            {
                matrix.SetCell(i, 0, 0.0f);
            }
        }
        return matrix;
    }

    public static int CheckMemory(bool[] input)
    {
        //Debug.Log("Because I'm confused, I jumped randomly 1");

        input0 = BoolArrayToMatrix(input);
        FeedForward();

        int maxim = 0;
        float maximVal = output2.GetCell(0,0);
        string predict = "";
        for (int i = 0;i < output2.GetRow(); i++)
        {
            if (output2.GetCell(i, 0) > maximVal)
            {
                maximVal = output2.GetCell(i, 0);
                maxim = i;
            }
            predict += output2.GetCell(i,0) + " , ";
        }

        Debug.Log(predict);

        //return Random.Range(0, GetInputCount());
        return maxim;
    }

    public static void AddMemory(bool[] input, int output, bool result)
    {
        if (result)
        {
            input0 = BoolArrayToMatrix(input);

            if (output == 0)
            {
                SetTargetOutput(new Matrix(new float[2, 1] { { 1 }, { 0 } }));
            } else if (output == 1)
            {
                SetTargetOutput(new Matrix(new float[2, 1] { { 0 }, { 1 } }));
            }

            ReadyTraining();
        } else
        {
            input0 = BoolArrayToMatrix(input);

            if (output == 0)
            {
                SetTargetOutput(new Matrix(new float[2, 1] { { 1 }, { 1 } }));
            }
            else if (output == 1)
            {
                SetTargetOutput(new Matrix(new float[2, 1] { { 0 }, { 0 } }));
            }

            ReadyTraining();
        }
    }

    public static void ReadyTraining()
    {
        //SetTargetOutput(new Matrix(new float[,] { { 0 }, { 1 } })); //Maybe ? not fixed;
        StartTraining();
    }

    public static void SetTargetOutput(Matrix matrix)
    {
        targetOutput = matrix;
    }

    public static void StartTraining()
    {
        FeedForward();
        CalculateError();
        CalculateGradient();
        CheckGradient();
        BackPropagation();
    }

    public static void FeedForward()
    {
        hidden1 = Matrix.Add(Matrix.Multiply(weight_0_1,input0) , Matrix.Multiply(bias_weight_1, bias_1));
        ActivateFunction(hidden1);

        output2 = Matrix.Add(Matrix.Multiply(weight_1_2, hidden1), Matrix.Multiply(bias_weight_2, bias_2));
        ActivateFunction(output2);

        Debug.Log("FeedForward Complete");
    }

    public static void CalculateError()
    {
        float temp = 0;

        for (int i = 0;i < targetOutput.GetRow(); i++)
        {
            for (int j = 0; j < targetOutput.GetColumn(); j++)
            {
                temp += Mathf.Pow(targetOutput.GetCell(i,j) - output2.GetCell(i,j),2);
            }
        }

        totalError = temp / 2.0f;

        Debug.Log("CalculateError Complete");
    }

    static float dw1 = 0;
    static float dw2 = 0;
    static float dw3 = 0;
    static float dw4 = 0;
    static float dw5 = 0;
    static float dw6 = 0;
    static float dw7 = 0;
    static float dw8 = 0;

    static float dbw1 = 0;
    static float dbw2 = 0;
    static float dbw3 = 0;
    static float dbw4 = 0;

    public static void CalculateGradient()
    {
        //Matrix deltaZ = GetDeltaZ();
        //DeltaZ = (z - t)z(1 - z)
        dw5 = CountDeltaZ(0, 0);
        dw6 = CountDeltaZ(1, 0);
        dw7 = CountDeltaZ(0, 1);
        dw8 = CountDeltaZ(1, 1);

        dbw3 = CountBaseDeltaZ(0);
        dbw4 = CountBaseDeltaZ(1);

        //Debug.Log(weight_1_2.GetCell(1, 1));
        //Debug.Log(hidden1.GetCell(0, 1));

        dw1 = (dbw3 * weight_1_2.GetCell(0, 0) + dbw4 * weight_1_2.GetCell(1, 0)) * hidden1.GetCell(0, 0) * (1 - hidden1.GetCell(0, 0)) * input0.GetCell(0, 0); //w5 and w6, outb1 and outa1
        dw2 = (dbw3 * weight_1_2.GetCell(0, 1) + dbw4 * weight_1_2.GetCell(1, 1)) * hidden1.GetCell(1, 0) * (1 - hidden1.GetCell(1, 0)) * input0.GetCell(0, 0); //w7 and w8, outb2 and outa1
        dw3 = (dbw3 * weight_1_2.GetCell(0, 0) + dbw4 * weight_1_2.GetCell(1, 0)) * hidden1.GetCell(0, 0) * (1 - hidden1.GetCell(0, 0)) * input0.GetCell(1, 0); //w5 and w6, outb1 and outa2
        dw4 = (dbw3 * weight_1_2.GetCell(0, 1) + dbw4 * weight_1_2.GetCell(1, 1)) * hidden1.GetCell(1, 0) * (1 - hidden1.GetCell(1, 0)) * input0.GetCell(1, 0); //w7 and w8, outb2 and outa2

        dbw1 = (dbw3 * weight_1_2.GetCell(0, 0) + dbw4 * weight_1_2.GetCell(1, 0)) * hidden1.GetCell(0, 0) * (1 - hidden1.GetCell(0, 0)); //w5 and w6, outb1
        dbw2 = (dbw3 * weight_1_2.GetCell(0, 1) + dbw4 * weight_1_2.GetCell(1, 1)) * hidden1.GetCell(1, 0) * (1 - hidden1.GetCell(1, 0)); //w7 and w8, outb2

        Debug.Log("CalculateGradient Complete");
    }

    public static Matrix GetDeltaZ()
    {
        Matrix a = Matrix.Substract(output2, targetOutput);
        Matrix b = output2;
        Matrix c = Matrix.OneSubstraction(output2);
        return Matrix.Times(Matrix.Times(a,b),c);
    }

    public static float CountBaseDeltaZ(int row1)
    {
        return ((output2.GetCell(row1, 0) - targetOutput.GetCell(row1, 0)) * output2.GetCell(row1, 0) * (1 - output2.GetCell(row1, 0)));
    }

    public static float CountDeltaZ(int row1, int row2)
    {
        return CountBaseDeltaZ(row1) * hidden1.GetCell(row2, 0);
    }

    public static void CheckGradient()
    {
        //Not yet, masih malas
    }

    public static void BackPropagation()
    {
        weight_1_2.SetCell(1, 1,weight_1_2.GetCell(1, 1) - learning_rate * dw8);
        weight_1_2.SetCell(0, 1, weight_1_2.GetCell(0, 1) - learning_rate * dw7);
        weight_1_2.SetCell(1, 0, weight_1_2.GetCell(1, 0) - learning_rate * dw6);
        weight_1_2.SetCell(0, 0, weight_1_2.GetCell(0, 0) - learning_rate * dw5);

        weight_0_1.SetCell(1, 1, weight_0_1.GetCell(1, 1) - learning_rate * dw4);
        weight_0_1.SetCell(0, 1, weight_0_1.GetCell(0, 1) - learning_rate * dw3);
        weight_0_1.SetCell(1, 0, weight_0_1.GetCell(1, 0) - learning_rate * dw2);
        weight_0_1.SetCell(0, 0, weight_0_1.GetCell(0, 0) - learning_rate * dw1);

        bias_weight_1.SetCell(0, 0, bias_weight_1.GetCell(0, 0) - learning_rate * dbw1);
        bias_weight_1.SetCell(1, 0, bias_weight_1.GetCell(1, 0) - learning_rate * dbw2);

        bias_weight_2.SetCell(0, 0, bias_weight_2.GetCell(0, 0) - learning_rate * dbw3);
        bias_weight_2.SetCell(1, 0, bias_weight_2.GetCell(1, 0) - learning_rate * dbw4);

        Debug.Log("BackPropagation Complete");
    }

    public static void ActivateFunction(Matrix matrix)
    {
        for (int i = 0;i < matrix.GetRow(); i++)
        {
            for (int j = 0;j < matrix.GetColumn(); j++)
            {
                matrix.SetCell(i,j,SigmoidFunction(matrix.GetCell(i,j)));
            }
        }
    }

    public static float SigmoidFunction(float x)
    {
        return 1.0f / (1 + Mathf.Exp(-x));
    }
}


public class Matrix
{
    private float[,] values;
    private int row = 0;
    private int column = 0;

    public Matrix(float[,] arr, bool isRandom = false)
    {
        values = arr;
        row = values.GetLength(0);
        column = values.GetLength(1);

        if (isRandom)
        {
            Randomize();
        }
    }

    public float GetCell(int rows, int columns)
    {
        return values[rows, columns];
    }

    public void SetCell(int rows, int columns, float x)
    {
        values[rows, columns] = x;
    }

    public int GetRow()
    {
        return row;
    }

    public int GetColumn()
    {
        return column;
    }

    public void Randomize()
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                values[i,j] = Random.Range(0.0f, 1.0f);
            }
        }
    }

    public override string ToString()
    {
        string output = "[";
        for (int i = 0; i < GetRow(); i++)
        {
            output += "[ ";
            for (int j = 0; j < GetColumn(); j++)
            {
                output += values[i, j];
                if (j < GetColumn() - 1)
                {
                    output += ", ";
                }
            }
            output += " ]\n";
        }
        output += " ],";
        return output;
    }

    public static Matrix Add(Matrix a, Matrix b)
    {
        if (a.GetColumn() == b.GetColumn() && a.GetRow() == b.GetRow())
        {
            int row = a.GetRow();
            int column = a.GetColumn();

            float[,] proc = new float[row, column];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    proc[i, j] = a.GetCell(i, j) + b.GetCell(i, j);
                }
            }

            Matrix matrix = new Matrix(proc);
            return matrix;
        }

        throw new UnityException("Row or Column not equal");
    }

    public static Matrix Substract(Matrix a, Matrix b)
    {
        if (a.GetColumn() == b.GetColumn() && a.GetRow() == b.GetRow())
        {
            int row = a.GetRow();
            int column = a.GetColumn();

            float[,] proc = new float[row, column];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    proc[i, j] = a.GetCell(i, j) - b.GetCell(i, j);
                }
            }

            Matrix matrix = new Matrix(proc);
            return matrix;
        }

        throw new UnityException("Row or Column not equal");
    }

    public static Matrix OneSubstraction(Matrix a)
    {
        int row = a.GetRow();
        int column = a.GetColumn();

        float[,] proc = new float[row, column];

        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                proc[i, j] = 1 - a.GetCell(i, j);
            }
        }

        Matrix matrix = new Matrix(proc);
        return matrix;
    }

    public static Matrix Times(Matrix a, Matrix b)
    {
        if (a.GetColumn() == b.GetColumn() && a.GetRow() == b.GetRow())
        {
            int row = a.GetRow();
            int column = a.GetColumn();

            float[,] proc = new float[row, column];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    proc[i, j] = a.GetCell(i, j) * b.GetCell(i, j);
                }
            }

            Matrix matrix = new Matrix(proc);
            return matrix;
        }

        throw new UnityException("Row or Column not equal");
    }

    public static Matrix Multiply(Matrix a, Matrix b)
    {
        if (a.GetColumn() == b.GetRow())
        {
            int row = a.GetRow();
            int column = b.GetColumn();
            int fork = b.GetRow();

            float[,] proc = new float[row, column];

            float sum = 0;

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    for (int k = 0; k < fork; k++)
                    {
                        sum += a.GetCell(i, k) * b.GetCell(k, j);
                    }
                    proc[i, j] = sum;
                    sum = 0;
                }
            }

            Matrix matrix = new Matrix(proc);
            return matrix;
        }

        throw new UnityException("Row or Column not good");
    }
}