﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FrogGamePlayer : MonoBehaviour
{
    public FrogGameMaster gameMaster;

    public float jumpForce;
    public float raycastDownRange;

    Rigidbody2D rigidbody2D;

    public static  bool isGround = false;

    public static bool CheckGround()
    {
        return isGround;
    }

    void Start()
    {
        rigidbody2D = GetComponent<Rigidbody2D>();
        isGround = false;
    }

    public void Jump()
    {
        if (!FrogGameMaster.isGameOver) {
            if (isGround)
            {
                rigidbody2D.velocity = new Vector2(0, jumpForce);
                transform.position += new Vector3(0,2,0);
                isGround = false;
            }
        }
    }

    bool firstOnce = false;

    void OnCollisionEnter2D(Collision2D other)
    {
        if (firstOnce)
        {
            if (!isGround)
            {
                if (other.gameObject.tag == "Terrain")
                {
                    if (!other.gameObject.GetComponent<FrogGameTerrain>().isSafeTerrain)
                    {
                        FrogGameMaster.isGameOver = true;
                        FrogGameNeuralNetwork.AddMemory((bool[]) FrogGameMaster.latestIndex.Clone(), FrogGameMaster.latestJumpType, false);
                    }
                    else
                    {
                        FrogGameNeuralNetwork.AddMemory((bool[])FrogGameMaster.latestIndex.Clone(), FrogGameMaster.latestJumpType, true);
                    }

                    gameMaster.GetLatestIndex();
                }
                isGround = true;
            }
        }
        else
        {
            firstOnce = true;
            isGround = true;
        }
    }

    void OnCollisionExit2D()
    {
        //isGround = false;
    }
}
