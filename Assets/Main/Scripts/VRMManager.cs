using UnityEngine;
using UniVRM10;
using System;
using System.IO;
using OpenSee;
using MediaPipeTracking;

/// <summary>
/// Manages VRM model loading and distributes the instance to tracking systems.
/// </summary>
public class VRMManager : MonoBehaviour
{
    [Header("Spawn Settings")]
    [Tooltip("Where the VRM model spawns relative to the camera")]
    public Vector3 spawnPosition = new Vector3(0f, -0.8f, 1.8f);

    [Header("Tracking Sync References")]
    public OpenSeeVRMSync openSeeVRMSync;
    public MediaPipeVRMSync mediaPipeVRMSync;

    public Vrm10Instance CurrentModel { get; private set; }

    public async void LoadVRM(string path)
    {
        try
        {
            // Destroy previous model if one exists
            if (CurrentModel != null)
                Destroy(CurrentModel.gameObject);

            byte[] bytes = File.ReadAllBytes(path);
            var vrmInstance = await Vrm10.LoadBytesAsync(bytes);

            vrmInstance.gameObject.transform.position = spawnPosition;
            CurrentModel = vrmInstance;

            if (openSeeVRMSync != null)
                openSeeVRMSync.vrmInstance = vrmInstance;

            if (mediaPipeVRMSync != null)
            {
                mediaPipeVRMSync.vrmInstance = vrmInstance;
                mediaPipeVRMSync.OnVRMLoaded();
            }
        }
        catch (Exception e)
        {
            Debug.LogError("VRM Import Error: " + e);
        }
    }
}
