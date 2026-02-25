using UnityEngine;
using UnityEngine.UI;
using SFB;
using System.IO;

public class UIManager : MonoBehaviour
{
    [Header("UI Buttons")]
    public Button loadVRMButton;

    [Header("Manager References")]
    public VRMManager vrmManager;
    public TrackingManager trackingManager;

    void Start()
    {
        loadVRMButton.onClick.AddListener(OnLoadVRMClicked);
    }

    void OnLoadVRMClicked()
    {
        var extensions = new[] {
            new ExtensionFilter("VRM Files", "vrm"),
        };

        StandaloneFileBrowser.OpenFilePanelAsync("Open VRM Model", "", extensions, false, paths =>
        {
            if (paths.Length > 0 && File.Exists(paths[0]))
            {
                vrmManager.LoadVRM(paths[0]);
            }
        });
    }

    public void OnTrackingSourceChanged(int index)
    {
        if (trackingManager != null)
            trackingManager.SetTrackingSource(index);
    }

    void OnQuitClicked()
    {
        Application.Quit();
    }
}
