using System.Collections;
using UnityEngine;
using UniVRM10;

namespace MediaPipeTracking
{
    /// <summary>
    /// Maps MediaPipe tracking data to VRM10 expressions and humanoid bone rotations.
    /// Supports 52 ARKit blendshapes → VRM expressions and 17 bone quaternions → humanoid.
    /// Face expressions are applied in Update(), bone rotations in LateUpdate()
    /// (after VRM's own LateUpdate processes).
    /// </summary>
    [DefaultExecutionOrder(20000)] // Run after VRM's LateUpdate
    public class MediaPipeVRMSync : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("The MediaPipe UDP receiver")]
        public MediaPipeReceiver receiver;

        [Tooltip("The VRM10 instance to drive")]
        public Vrm10Instance vrmInstance;

        [Header("Face Tracking")]
        public bool enableFaceTracking = true;

        [Range(0f, 1f)]
        [Tooltip("Minimum face confidence to apply expressions")]
        public float faceConfidenceThreshold = 0.3f;

        [Range(0f, 1f)]
        [Tooltip("Temporal smoothing for face blendshapes (0 = no smoothing, 1 = max)")]
        public float faceSmoothing = 0.4f;

        [Range(0.5f, 3f)]
        [Tooltip("Global multiplier for expression intensity")]
        public float expressionMultiplier = 1.2f;

        [Header("Body Tracking")]
        public bool enableBodyTracking = true;

        [Range(0f, 1f)]
        [Tooltip("Minimum body confidence to apply bone rotations")]
        public float bodyConfidenceThreshold = 0.3f;

        [Range(0f, 0.95f)]
        [Tooltip("Motion smoothing — interpolates between frames for buttery movement at low tracking FPS. 0 = raw (snappy), 0.5 = balanced, 0.9 = very smooth")]
        public float motionSmoothing = 0.5f;

        [Range(1f, 2f)]
        [Tooltip("Arm rotation amplification to compensate for MediaPipe landmark imprecision")]
        public float armAmplification = 1.4f;

        // VRM Expression keys
        private ExpressionKey blinkLeftKey;
        private ExpressionKey blinkRightKey;
        private ExpressionKey lookUpKey;
        private ExpressionKey lookDownKey;
        private ExpressionKey lookLeftKey;
        private ExpressionKey lookRightKey;
        private ExpressionKey aaKey;
        private ExpressionKey ihKey;
        private ExpressionKey ouKey;
        private ExpressionKey eeKey;
        private ExpressionKey ohKey;
        private ExpressionKey happyKey;
        private ExpressionKey angryKey;
        private ExpressionKey sadKey;
        private ExpressionKey surprisedKey;

        // Bone transform caches
        private Transform[] boneTransforms = new Transform[MediaPipeData.NumBones];
        private Quaternion[] initialBoneRotations = new Quaternion[MediaPipeData.NumBones];
        private Quaternion[] smoothedBoneRotations = new Quaternion[MediaPipeData.NumBones];
        private bool bonesInitialized = false;

        // Smoothed expression values
        private float[] smoothedBlendshapes = new float[MediaPipeData.NumBlendshapes];

        // Current frame data (captured in Update, applied in LateUpdate)
        private MediaPipeData currentData;
        private bool hasNewData = false;

        private bool modelInitialized = false;

        void Start()
        {
            if (receiver == null)
                receiver = GetComponent<MediaPipeReceiver>();

            SetupExpressionKeys();

            if (vrmInstance != null)
                StartCoroutine(InitializeModel());
        }

        private void SetupExpressionKeys()
        {
            blinkLeftKey = ExpressionKey.CreateFromPreset(ExpressionPreset.blinkLeft);
            blinkRightKey = ExpressionKey.CreateFromPreset(ExpressionPreset.blinkRight);
            lookUpKey = ExpressionKey.CreateFromPreset(ExpressionPreset.lookUp);
            lookDownKey = ExpressionKey.CreateFromPreset(ExpressionPreset.lookDown);
            lookLeftKey = ExpressionKey.CreateFromPreset(ExpressionPreset.lookLeft);
            lookRightKey = ExpressionKey.CreateFromPreset(ExpressionPreset.lookRight);
            aaKey = ExpressionKey.CreateFromPreset(ExpressionPreset.aa);
            ihKey = ExpressionKey.CreateFromPreset(ExpressionPreset.ih);
            ouKey = ExpressionKey.CreateFromPreset(ExpressionPreset.ou);
            eeKey = ExpressionKey.CreateFromPreset(ExpressionPreset.ee);
            ohKey = ExpressionKey.CreateFromPreset(ExpressionPreset.oh);
            happyKey = ExpressionKey.CreateFromPreset(ExpressionPreset.happy);
            angryKey = ExpressionKey.CreateFromPreset(ExpressionPreset.angry);
            sadKey = ExpressionKey.CreateFromPreset(ExpressionPreset.sad);
            surprisedKey = ExpressionKey.CreateFromPreset(ExpressionPreset.surprised);
        }

        private IEnumerator InitializeModel()
        {
            yield return null; // Wait one frame for VRM to initialize

            if (vrmInstance != null && vrmInstance.Humanoid != null)
            {
                CacheBoneTransforms();
                modelInitialized = true;
                Debug.Log("[MediaPipeVRMSync] Model initialized with bone caching.");
            }
        }

        private void CacheBoneTransforms()
        {
            var humanoid = vrmInstance.Humanoid;

            boneTransforms[MediaPipeData.BONE_HIPS] = humanoid.Hips;
            boneTransforms[MediaPipeData.BONE_SPINE] = humanoid.Spine;
            boneTransforms[MediaPipeData.BONE_CHEST] = humanoid.Chest;
            boneTransforms[MediaPipeData.BONE_NECK] = humanoid.Neck;
            boneTransforms[MediaPipeData.BONE_HEAD] = humanoid.Head;
            boneTransforms[MediaPipeData.BONE_LEFT_SHOULDER] = humanoid.LeftShoulder;
            boneTransforms[MediaPipeData.BONE_LEFT_UPPER_ARM] = humanoid.LeftUpperArm;
            boneTransforms[MediaPipeData.BONE_LEFT_LOWER_ARM] = humanoid.LeftLowerArm;
            boneTransforms[MediaPipeData.BONE_LEFT_HAND] = humanoid.LeftHand;
            boneTransforms[MediaPipeData.BONE_RIGHT_SHOULDER] = humanoid.RightShoulder;
            boneTransforms[MediaPipeData.BONE_RIGHT_UPPER_ARM] = humanoid.RightUpperArm;
            boneTransforms[MediaPipeData.BONE_RIGHT_LOWER_ARM] = humanoid.RightLowerArm;
            boneTransforms[MediaPipeData.BONE_RIGHT_HAND] = humanoid.RightHand;
            boneTransforms[MediaPipeData.BONE_LEFT_UPPER_LEG] = humanoid.LeftUpperLeg;
            boneTransforms[MediaPipeData.BONE_LEFT_LOWER_LEG] = humanoid.LeftLowerLeg;
            boneTransforms[MediaPipeData.BONE_RIGHT_UPPER_LEG] = humanoid.RightUpperLeg;
            boneTransforms[MediaPipeData.BONE_RIGHT_LOWER_LEG] = humanoid.RightLowerLeg;

            // Store initial local rotations (T-pose) and init smoothed state
            for (int i = 0; i < MediaPipeData.NumBones; i++)
            {
                if (boneTransforms[i] != null)
                {
                    initialBoneRotations[i] = boneTransforms[i].localRotation;
                    smoothedBoneRotations[i] = boneTransforms[i].localRotation;
                }
                else
                {
                    initialBoneRotations[i] = Quaternion.identity;
                    smoothedBoneRotations[i] = Quaternion.identity;
                }
            }

            bonesInitialized = true;
        }

        void Update()
        {
            if (receiver == null || vrmInstance == null)
                return;

            MediaPipeData data = receiver.LatestData;
            if (data == null)
                return;

            // Re-initialize bones if VRM instance changed
            if (!modelInitialized || !bonesInitialized)
            {
                if (vrmInstance.Humanoid != null)
                {
                    CacheBoneTransforms();
                    modelInitialized = true;
                }
                else
                {
                    return;
                }
            }

            // Grab latest data for this frame
            currentData = data;
            hasNewData = true;

            // Face expressions can be set in Update — VRM reads them in LateUpdate
            if (enableFaceTracking && data.faceConfidence >= faceConfidenceThreshold)
                UpdateFaceExpressions(data);
        }

        void LateUpdate()
        {
            if (!hasNewData || currentData == null)
                return;

            // Apply bone rotations AFTER VRM's own LateUpdate has run
            if (enableBodyTracking && currentData.bodyConfidence >= bodyConfidenceThreshold && bonesInitialized)
                UpdateBoneRotations(currentData);
        }

        private void UpdateFaceExpressions(MediaPipeData data)
        {
            if (vrmInstance.Runtime == null || vrmInstance.Runtime.Expression == null)
                return;

            var bs = data.blendshapes;
            // Frame-rate-independent smoothing for face
            float t = 1f - Mathf.Pow(faceSmoothing, Time.deltaTime * 60f);

            // Smooth all blendshapes
            for (int i = 0; i < MediaPipeData.NumBlendshapes; i++)
                smoothedBlendshapes[i] = Mathf.Lerp(smoothedBlendshapes[i], bs[i], t);

            float S(int idx) => smoothedBlendshapes[idx];
            float Avg(int a, int b) => (smoothedBlendshapes[a] + smoothedBlendshapes[b]) * 0.5f;
            var expr = vrmInstance.Runtime.Expression;

            // --- Eyes ---
            expr.SetWeight(blinkLeftKey, Mathf.Clamp01(S(MediaPipeData.BS_EYE_BLINK_LEFT)));
            expr.SetWeight(blinkRightKey, Mathf.Clamp01(S(MediaPipeData.BS_EYE_BLINK_RIGHT)));

            expr.SetWeight(lookUpKey, Mathf.Clamp01(Avg(MediaPipeData.BS_EYE_LOOK_UP_LEFT, MediaPipeData.BS_EYE_LOOK_UP_RIGHT)));
            expr.SetWeight(lookDownKey, Mathf.Clamp01(Avg(MediaPipeData.BS_EYE_LOOK_DOWN_LEFT, MediaPipeData.BS_EYE_LOOK_DOWN_RIGHT)));
            expr.SetWeight(lookLeftKey, Mathf.Clamp01(Avg(MediaPipeData.BS_EYE_LOOK_OUT_LEFT, MediaPipeData.BS_EYE_LOOK_IN_RIGHT)));
            expr.SetWeight(lookRightKey, Mathf.Clamp01(Avg(MediaPipeData.BS_EYE_LOOK_OUT_RIGHT, MediaPipeData.BS_EYE_LOOK_IN_LEFT)));

            // --- Mouth ---
            // MediaPipe jawOpen has a ~0.3 baseline when mouth is closed.
            // mouthClose counteracts it. Subtract mouthClose and apply a deadzone.
            float rawJaw = S(MediaPipeData.BS_JAW_OPEN);
            float mouthClose = smoothedBlendshapes[26]; // mouthClose index
            float jawOpen = Mathf.Clamp01((rawJaw - mouthClose - 0.1f) * 2.0f);

            float pucker = S(MediaPipeData.BS_MOUTH_PUCKER);
            float funnel = S(MediaPipeData.BS_MOUTH_FUNNEL);
            float smile = Avg(MediaPipeData.BS_MOUTH_SMILE_LEFT, MediaPipeData.BS_MOUTH_SMILE_RIGHT);
            float stretch = Avg(MediaPipeData.BS_MOUTH_STRETCH_LEFT, MediaPipeData.BS_MOUTH_STRETCH_RIGHT);

            // Vowels are mutually exclusive — pick the dominant one
            float aa = jawOpen;                                          // mouth wide open
            float ou = Mathf.Min(funnel + pucker * 0.5f, 1f);           // rounded lips
            float ih = stretch * 0.7f;                                   // wide/teeth
            float ee = (jawOpen < 0.1f) ? smile : 0f;                   // smile closed
            float oh = Mathf.Min(pucker * 0.5f + jawOpen * 0.5f, 1f);   // rounded open

            // Find the dominant vowel and only apply that one
            float maxVowel = Mathf.Max(Mathf.Max(aa, ou), Mathf.Max(Mathf.Max(ih, ee), oh));

            expr.SetWeight(aaKey, (aa >= maxVowel - 0.01f && maxVowel > 0.05f) ? Mathf.Clamp01(aa * expressionMultiplier) : 0f);
            expr.SetWeight(ouKey, (ou >= maxVowel - 0.01f && maxVowel > 0.05f && aa < ou) ? Mathf.Clamp01(ou * expressionMultiplier) : 0f);
            expr.SetWeight(ihKey, (ih >= maxVowel - 0.01f && maxVowel > 0.05f && aa < ih) ? Mathf.Clamp01(ih * expressionMultiplier) : 0f);
            expr.SetWeight(eeKey, Mathf.Clamp01(ee * expressionMultiplier));
            expr.SetWeight(ohKey, (oh >= maxVowel - 0.01f && maxVowel > 0.05f && aa < oh && ou < oh) ? Mathf.Clamp01(oh * expressionMultiplier) : 0f);

            // --- Emotional expressions: only apply if clearly dominant ---
            float happy = smile;
            float angry = Avg(MediaPipeData.BS_BROW_DOWN_LEFT, MediaPipeData.BS_BROW_DOWN_RIGHT);
            float sad = Avg(MediaPipeData.BS_MOUTH_FROWN_LEFT, MediaPipeData.BS_MOUTH_FROWN_RIGHT);
            float surprised = Avg(MediaPipeData.BS_EYE_WIDE_LEFT, MediaPipeData.BS_EYE_WIDE_RIGHT) * 0.5f
                            + smoothedBlendshapes[MediaPipeData.BS_BROW_INNER_UP] * 0.5f;

            // Only show emotion if it's strong enough (threshold 0.3)
            float emotionThreshold = 0.3f;
            expr.SetWeight(happyKey, happy > emotionThreshold ? Mathf.Clamp01((happy - emotionThreshold) * 1.5f) : 0f);
            expr.SetWeight(angryKey, angry > emotionThreshold ? Mathf.Clamp01((angry - emotionThreshold) * 1.5f) : 0f);
            expr.SetWeight(sadKey, sad > emotionThreshold ? Mathf.Clamp01((sad - emotionThreshold) * 1.5f) : 0f);
            expr.SetWeight(surprisedKey, surprised > emotionThreshold ? Mathf.Clamp01((surprised - emotionThreshold) * 1.5f) : 0f);
        }

        private void UpdateBoneRotations(MediaPipeData data)
        {
            // Frame-rate-independent smoothing:
            // At 60fps with motionSmoothing=0.5, each frame blends ~50% toward target.
            // This makes 19fps tracking data look smooth at any render framerate.
            float smoothFactor = 1f - Mathf.Pow(motionSmoothing, Time.deltaTime * 60f);

            for (int i = 0; i < MediaPipeData.NumBones; i++)
            {
                if (boneTransforms[i] == null)
                    continue;

                Quaternion packetRot = data.boneRotations[i];

                // Amplify arm rotations to compensate for MediaPipe landmark imprecision
                bool isUpperArm = i == MediaPipeData.BONE_LEFT_UPPER_ARM || i == MediaPipeData.BONE_RIGHT_UPPER_ARM;
                bool isLowerArm = i == MediaPipeData.BONE_LEFT_LOWER_ARM || i == MediaPipeData.BONE_RIGHT_LOWER_ARM;
                if (isUpperArm || isLowerArm)
                {
                    float angle;
                    Vector3 axis;
                    packetRot.ToAngleAxis(out angle, out axis);
                    if (angle > 0.5f)
                        packetRot = Quaternion.AngleAxis(angle * armAmplification, axis);
                }

                Quaternion targetRotation = initialBoneRotations[i] * packetRot;

                // Slerp from our own tracked state, NOT from boneTransform.localRotation
                // (VRM's LateUpdate resets bones before we run, so reading from the
                // transform would oscillate between VRM's rest pose and our target)
                smoothedBoneRotations[i] = Quaternion.Slerp(
                    smoothedBoneRotations[i],
                    targetRotation,
                    smoothFactor
                );
                boneTransforms[i].localRotation = smoothedBoneRotations[i];
            }
        }

        /// <summary>
        /// Called when a new VRM model is loaded. Re-initializes bone caching.
        /// </summary>
        public void OnVRMLoaded()
        {
            modelInitialized = false;
            bonesInitialized = false;
            currentData = null;
            hasNewData = false;

            // Reset smoothed values
            for (int i = 0; i < MediaPipeData.NumBlendshapes; i++)
                smoothedBlendshapes[i] = 0f;

            if (vrmInstance != null && isActiveAndEnabled)
                StartCoroutine(InitializeModel());
        }
    }
}
