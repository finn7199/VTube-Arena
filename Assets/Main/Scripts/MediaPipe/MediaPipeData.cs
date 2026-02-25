using UnityEngine;

namespace MediaPipeTracking
{
    /// <summary>
    /// Data container for a single MediaPipe tracking frame.
    /// Parsed from the binary UDP packet sent by mediapipe_tracker.py.
    /// </summary>
    public class MediaPipeData
    {
        // --- Packet header ---
        public uint sequenceNumber;
        public float timestamp;

        // --- Face tracking: 52 ARKit blendshapes ---
        public float[] blendshapes = new float[NumBlendshapes];

        // --- Body tracking: 17 bone local rotations ---
        public Quaternion[] boneRotations = new Quaternion[NumBones];

        // --- Hip world position ---
        public Vector3 hipPosition;

        // --- Confidence scores ---
        public float faceConfidence;
        public float bodyConfidence;

        // --- Constants ---
        public const int NumBlendshapes = 52;
        public const int NumBones = 17;
        public const uint Magic = 0x4D504654; // "MPFT"
        public const int PacketSize = 512;

        // Bone indices
        public const int BONE_HIPS = 0;
        public const int BONE_SPINE = 1;
        public const int BONE_CHEST = 2;
        public const int BONE_NECK = 3;
        public const int BONE_HEAD = 4;
        public const int BONE_LEFT_SHOULDER = 5;
        public const int BONE_LEFT_UPPER_ARM = 6;
        public const int BONE_LEFT_LOWER_ARM = 7;
        public const int BONE_LEFT_HAND = 8;
        public const int BONE_RIGHT_SHOULDER = 9;
        public const int BONE_RIGHT_UPPER_ARM = 10;
        public const int BONE_RIGHT_LOWER_ARM = 11;
        public const int BONE_RIGHT_HAND = 12;
        public const int BONE_LEFT_UPPER_LEG = 13;
        public const int BONE_LEFT_LOWER_LEG = 14;
        public const int BONE_RIGHT_UPPER_LEG = 15;
        public const int BONE_RIGHT_LOWER_LEG = 16;

        // ARKit blendshape names in packet order
        public static readonly string[] BlendshapeNames = new string[]
        {
            "browDownLeft",         // 0
            "browDownRight",        // 1
            "browInnerUp",          // 2
            "browOuterUpLeft",      // 3
            "browOuterUpRight",     // 4
            "cheekPuff",            // 5
            "cheekSquintLeft",      // 6
            "cheekSquintRight",     // 7
            "eyeBlinkLeft",         // 8
            "eyeBlinkRight",        // 9
            "eyeLookDownLeft",      // 10
            "eyeLookDownRight",     // 11
            "eyeLookInLeft",        // 12
            "eyeLookInRight",       // 13
            "eyeLookOutLeft",       // 14
            "eyeLookOutRight",      // 15
            "eyeLookUpLeft",        // 16
            "eyeLookUpRight",       // 17
            "eyeSquintLeft",        // 18
            "eyeSquintRight",       // 19
            "eyeWideLeft",          // 20
            "eyeWideRight",         // 21
            "jawForward",           // 22
            "jawLeft",              // 23
            "jawOpen",              // 24
            "jawRight",             // 25
            "mouthClose",           // 26
            "mouthDimpleLeft",      // 27
            "mouthDimpleRight",     // 28
            "mouthFrownLeft",       // 29
            "mouthFrownRight",      // 30
            "mouthFunnel",          // 31
            "mouthLeft",            // 32
            "mouthLowerDownLeft",   // 33
            "mouthLowerDownRight",  // 34
            "mouthPressLeft",       // 35
            "mouthPressRight",      // 36
            "mouthPucker",          // 37
            "mouthRight",           // 38
            "mouthRollLower",       // 39
            "mouthRollUpper",       // 40
            "mouthShrugLower",      // 41
            "mouthShrugUpper",      // 42
            "mouthSmileLeft",       // 43
            "mouthSmileRight",      // 44
            "mouthStretchLeft",     // 45
            "mouthStretchRight",    // 46
            "mouthUpperUpLeft",     // 47
            "mouthUpperUpRight",    // 48
            "noseSneerLeft",        // 49
            "noseSneerRight",       // 50
            "tongueOut",            // 51
        };

        // Blendshape index constants for convenient access
        public const int BS_BROW_DOWN_LEFT = 0;
        public const int BS_BROW_DOWN_RIGHT = 1;
        public const int BS_BROW_INNER_UP = 2;
        public const int BS_CHEEK_PUFF = 5;
        public const int BS_EYE_BLINK_LEFT = 8;
        public const int BS_EYE_BLINK_RIGHT = 9;
        public const int BS_EYE_LOOK_DOWN_LEFT = 10;
        public const int BS_EYE_LOOK_DOWN_RIGHT = 11;
        public const int BS_EYE_LOOK_IN_LEFT = 12;
        public const int BS_EYE_LOOK_IN_RIGHT = 13;
        public const int BS_EYE_LOOK_OUT_LEFT = 14;
        public const int BS_EYE_LOOK_OUT_RIGHT = 15;
        public const int BS_EYE_LOOK_UP_LEFT = 16;
        public const int BS_EYE_LOOK_UP_RIGHT = 17;
        public const int BS_EYE_WIDE_LEFT = 20;
        public const int BS_EYE_WIDE_RIGHT = 21;
        public const int BS_JAW_OPEN = 24;
        public const int BS_MOUTH_FROWN_LEFT = 29;
        public const int BS_MOUTH_FROWN_RIGHT = 30;
        public const int BS_MOUTH_FUNNEL = 31;
        public const int BS_MOUTH_PUCKER = 37;
        public const int BS_MOUTH_SMILE_LEFT = 43;
        public const int BS_MOUTH_SMILE_RIGHT = 44;
        public const int BS_MOUTH_STRETCH_LEFT = 45;
        public const int BS_MOUTH_STRETCH_RIGHT = 46;
    }
}
