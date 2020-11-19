﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MemoryFragment
{
    public int input0 = 0;
    public int input1 = 0;

    public int output = 0;
    
    public bool result = false;

    public MemoryFragment(int input0, int input1, int output, bool result)
    {
        this.input0 = input0;
        this.input1 = input1;
        this.output = output;
        this.result = result;
    }

    public override string ToString()
    {
        return "Based On my experience " +
            input0.ToString() + " and " + input1.ToString() +
            " : " + output.ToString() + " is resulting in " + result.ToString();
    }
}


public static class FrogGameMemory
{
    public static List<MemoryFragment> memoryFragments = new List<MemoryFragment>();

    static FrogGameMemory()
    {
        memoryFragments = new List<MemoryFragment>();
    }

    public static int CheckMemory(int input0, int input1)
    {
        if (memoryFragments != null)
        {
            return SummarizeMemory(input0, input1);
        }

        Debug.Log("Because I'm confused, I jumped randomly 1");
        return Random.Range(0,2);
    }

    public static void AddMemory(int input0, int input1, int output, bool result)
    {
        if (memoryFragments != null && SearchEqualMemory(input0, input1) == false)
        {
            memoryFragments.Add(new MemoryFragment(input0, input1, output, result));
            Debug.Log(memoryFragments[memoryFragments.Count - 1]);
        }
        else
        {
            Debug.Log("Memory is exists, if I recall it again : " +
                new MemoryFragment(input0, input1, output, result)); //Useless Memory
        }
    }

    public static int SummarizeMemory(int input0, int input1)
    {
        MemoryFragment mfnew = new MemoryFragment(input0, input1, -1, false);
        foreach (MemoryFragment mf in memoryFragments)
        {
            int reason = ReasoningMemory(mf, mfnew);
            if (reason == -1)
            {

            } else
            {
                return reason;
            }
        }
        Debug.Log("Because I'm confused, I jumped randomly 2");
        return Random.Range(0, 2);
    }

    public static int ReasoningMemory(MemoryFragment experience, MemoryFragment moment)
    {
        if (experience.input0 == moment.input0 && experience.input1 == moment.input1){
            if (experience.result) //jika benar lakukan aja
            {
                return experience.output;
            } else //jika pengalman salah, harus diperbaiki
            {
                return experience.output == 1 ? 0 : 1;
            }
        }
        Debug.Log("Memory not found, I must think further");
        return -1;
    }

    public static bool SearchEqualMemory(int input0, int input1)
    {
        for (int i = 0; i < memoryFragments.Count; i++)
        {
            if (input0 == memoryFragments[i].input0 && input1 == memoryFragments[i].input1)
            {
                return true;
            }
        }
        return false;
    }
}
