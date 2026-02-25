using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine;

namespace MediaPipeTracking
{
    /// <summary>
    /// Launches and manages the MediaPipe Python tracker process.
    /// Mirrors OpenSeeLauncher pattern with simplified interface for Python scripts.
    /// </summary>
    public class MediaPipeLauncher : MonoBehaviour
    {
        [Header("Python Settings")]
        [Tooltip("Path to Python executable")]
        public string pythonPath = "python";

        [Tooltip("Camera device index")]
        public int cameraIndex = 0;

        [Header("Network")]
        [Tooltip("IP address to send tracking data to")]
        public string targetIP = "127.0.0.1";

        [Tooltip("UDP port to send tracking data to")]
        public int targetPort = 11574;

        [Header("Tracking Options")]
        [Tooltip("Disable body tracking (face only)")]
        public bool disableBody = false;

        [Tooltip("Disable face tracking (body only)")]
        public bool disableFace = false;

        [Tooltip("Camera capture width")]
        public int cameraWidth = 640;

        [Tooltip("Camera capture height")]
        public int cameraHeight = 480;

        [Tooltip("Show camera preview window with landmark overlay (for debugging)")]
        public bool showCameraPreview = false;

        [Header("Process Settings")]
        [Tooltip("Start tracker automatically on scene load")]
        public bool autoStart = false;

        [Tooltip("Log the command line used to start the tracker")]
        public bool logCommandline = false;

        [Header("Runtime Info")]
        [Tooltip("Whether the tracker process is currently running")]
        public bool trackerAlive = false;

        private Process trackerProcess = null;
        private StringBuilder outputBuffer = null;
        private OpenSee.Job job = null;

        void Start()
        {
            if (autoStart)
                StartTracker();
        }

        void OnEnable()
        {
            if (autoStart && !trackerAlive)
                StartTracker();
        }

        void Update()
        {
            if (trackerProcess != null)
            {
                try
                {
                    trackerAlive = !trackerProcess.HasExited;
                }
                catch
                {
                    trackerAlive = false;
                }
            }
            else
            {
                trackerAlive = false;
            }

            // Print buffered output
            if (outputBuffer != null && outputBuffer.Length > 0)
            {
                string output = outputBuffer.ToString();
                UnityEngine.Debug.Log("[MediaPipe] " + output);
                outputBuffer.Clear();
            }
        }

        public bool StartTracker()
        {
            if (job == null)
                job = new OpenSee.Job();

            string scriptPath = GetScriptPath();
            if (!File.Exists(scriptPath))
            {
                UnityEngine.Debug.LogError($"[MediaPipeLauncher] Script not found: {scriptPath}");
                return false;
            }

            // Build arguments
            StringBuilder args = new StringBuilder();
            args.Append($"\"{scriptPath}\"");
            args.Append($" --camera {cameraIndex}");
            args.Append($" --port {targetPort}");
            args.Append($" --ip {targetIP}");
            args.Append($" --width {cameraWidth}");
            args.Append($" --height {cameraHeight}");

            if (disableBody)
                args.Append(" --no-body");
            if (disableFace)
                args.Append(" --no-face");
            if (showCameraPreview)
                args.Append(" --show-camera");

            string argumentString = args.ToString();

            if (logCommandline)
                UnityEngine.Debug.Log($"[MediaPipeLauncher] Starting: {pythonPath} {argumentString}");

            StopTracker();

            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.CreateNoWindow = true;
                startInfo.UseShellExecute = false;
                startInfo.RedirectStandardOutput = true;
                startInfo.RedirectStandardError = true;
                startInfo.RedirectStandardInput = true;
                startInfo.FileName = pythonPath;
                startInfo.Arguments = argumentString;
                startInfo.WorkingDirectory = Path.GetDirectoryName(scriptPath);

                outputBuffer = new StringBuilder();
                trackerProcess = new Process();
                trackerProcess.StartInfo = startInfo;
                trackerProcess.EnableRaisingEvents = true;

                trackerProcess.OutputDataReceived += (sender, e) =>
                {
                    if (e.Data != null)
                        outputBuffer.AppendLine(e.Data);
                };
                trackerProcess.ErrorDataReceived += (sender, e) =>
                {
                    if (e.Data != null)
                        outputBuffer.AppendLine(e.Data);
                };

                trackerProcess.Start();
                job.AddProcess(trackerProcess.Handle);
                trackerProcess.BeginOutputReadLine();
                trackerProcess.BeginErrorReadLine();

                trackerAlive = !trackerProcess.HasExited;
                return trackerAlive;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[MediaPipeLauncher] Failed to start tracker: {e.Message}");
                trackerProcess = null;
                return false;
            }
        }

        public void StopTracker()
        {
            if (trackerProcess != null)
            {
                try
                {
                    if (!trackerProcess.HasExited)
                    {
                        trackerProcess.CloseMainWindow();
                        trackerProcess.Kill();
                    }
                    trackerProcess.Close();
                }
                catch { }
                trackerProcess = null;
            }
            trackerAlive = false;
        }

        private string GetScriptPath()
        {
            return Path.Combine(Application.streamingAssetsPath, "MediaPipe", "mediapipe_tracker.py");
        }

        private void CleanJob()
        {
            if (job != null)
            {
                job.Dispose();
                job = null;
            }
        }

        void OnDestroy()
        {
            StopTracker();
            CleanJob();
        }

        void OnApplicationQuit()
        {
            StopTracker();
            CleanJob();
        }
    }
}
