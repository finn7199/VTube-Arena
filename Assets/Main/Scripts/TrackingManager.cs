using UnityEngine;
using OpenSee;
using MediaPipeTracking;

/// <summary>
/// Toggle system for switching between OpenSeeFace and MediaPipe tracking at runtime.
/// Enables/disables the appropriate component sets when the active source changes.
/// </summary>
public class TrackingManager : MonoBehaviour
{
    public enum TrackingSource
    {
        OpenSeeFace,
        MediaPipe
    }

    [Header("Active Tracking Source")]
    public TrackingSource activeSource = TrackingSource.OpenSeeFace;

    [Header("OpenSeeFace Components")]
    public OpenSee.OpenSee openSeeReceiver;
    public OpenSeeLauncher openSeeLauncher;
    public OpenSeeVRMSync openSeeVRMSync;

    [Header("MediaPipe Components")]
    public MediaPipeReceiver mediaPipeReceiver;
    public MediaPipeLauncher mediaPipeLauncher;
    public MediaPipeVRMSync mediaPipeVRMSync;

    private TrackingSource lastAppliedSource;

    void Start()
    {
        ApplyTrackingSource();
        lastAppliedSource = activeSource;
    }

    void Update()
    {
        if (activeSource != lastAppliedSource)
        {
            ApplyTrackingSource();
            lastAppliedSource = activeSource;
        }
    }

    /// <summary>
    /// Applies the current activeSource setting by enabling/disabling component sets.
    /// </summary>
    public void ApplyTrackingSource()
    {
        bool useOpenSee = activeSource == TrackingSource.OpenSeeFace;
        bool useMediaPipe = activeSource == TrackingSource.MediaPipe;

        // Stop the deactivated tracker's process
        if (!useOpenSee && openSeeLauncher != null)
            openSeeLauncher.StopTracker();
        if (!useMediaPipe && mediaPipeLauncher != null)
            mediaPipeLauncher.StopTracker();

        // Enable/disable OpenSee components
        if (openSeeReceiver != null)
            openSeeReceiver.enabled = useOpenSee;
        if (openSeeLauncher != null)
            openSeeLauncher.enabled = useOpenSee;
        if (openSeeVRMSync != null)
            openSeeVRMSync.enabled = useOpenSee;

        // Enable/disable MediaPipe components
        if (mediaPipeReceiver != null)
            mediaPipeReceiver.enabled = useMediaPipe;
        if (mediaPipeLauncher != null)
            mediaPipeLauncher.enabled = useMediaPipe;
        if (mediaPipeVRMSync != null)
            mediaPipeVRMSync.enabled = useMediaPipe;

        Debug.Log($"[TrackingManager] Switched to {activeSource}");
    }

    /// <summary>
    /// Switch tracking source. Can be called from UI dropdown.
    /// </summary>
    public void SetTrackingSource(int index)
    {
        activeSource = (TrackingSource)index;
    }
}
