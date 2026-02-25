using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace MediaPipeTracking
{
    /// <summary>
    /// UDP receiver for MediaPipe tracking data.
    /// Mirrors the OpenSee.cs pattern: background thread receives binary packets,
    /// parses them, and exposes the latest data via atomic reference swap.
    /// </summary>
    public class MediaPipeReceiver : MonoBehaviour
    {
        [Header("UDP Settings")]
        [Tooltip("Listen address. Use 0.0.0.0 for remote connections.")]
        public string listenAddress = "127.0.0.1";

        [Tooltip("UDP port to listen on (distinct from OpenSee's 11573)")]
        public int listenPort = 11574;

        [Header("Runtime Info")]
        [Tooltip("Number of valid packets received")]
        public int receivedPackets = 0;

        public bool listening { get; private set; } = false;

        /// <summary>
        /// The most recent tracking data. Thread-safe via reference swap.
        /// </summary>
        public MediaPipeData LatestData { get; private set; }

        private Socket socket;
        private byte[] buffer;
        private Thread receiveThread;
        private volatile bool stopReception = false;

        void Start()
        {
            buffer = new byte[65535];

            if (socket == null)
            {
                socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                IPAddress ip;
                IPAddress.TryParse(listenAddress, out ip);
                socket.Bind(new IPEndPoint(ip, listenPort));
                socket.ReceiveTimeout = 15;
            }

            receiveThread = new Thread(PerformReception);
            receiveThread.Start();
        }

        void Update()
        {
            if (receiveThread != null && !receiveThread.IsAlive)
            {
                Start();
            }
        }

        private void PerformReception()
        {
            EndPoint senderRemote = new IPEndPoint(IPAddress.Any, 0);
            listening = true;

            while (!stopReception)
            {
                try
                {
                    int receivedBytes = socket.ReceiveFrom(buffer, SocketFlags.None, ref senderRemote);

                    if (receivedBytes < MediaPipeData.PacketSize)
                        continue;

                    MediaPipeData data = ParsePacket(buffer, 0);
                    if (data != null)
                    {
                        LatestData = data; // Atomic reference swap
                        receivedPackets++;
                    }
                }
                catch (SocketException)
                {
                    // Timeout or other socket errors - continue
                }
                catch (Exception)
                {
                    // Unexpected error - continue
                }
            }
        }

        private MediaPipeData ParsePacket(byte[] b, int offset)
        {
            int o = offset;

            // Validate magic number
            uint magic = BitConverter.ToUInt32(b, o);
            o += 4;
            if (magic != MediaPipeData.Magic)
                return null;

            MediaPipeData data = new MediaPipeData();

            // Header
            data.sequenceNumber = BitConverter.ToUInt32(b, o);
            o += 4;
            data.timestamp = BitConverter.ToSingle(b, o);
            o += 4;

            // Blendshapes (52 floats)
            for (int i = 0; i < MediaPipeData.NumBlendshapes; i++)
            {
                data.blendshapes[i] = BitConverter.ToSingle(b, o);
                o += 4;
            }

            // Bone rotations (17 quaternions: x, y, z, w)
            for (int i = 0; i < MediaPipeData.NumBones; i++)
            {
                float x = BitConverter.ToSingle(b, o); o += 4;
                float y = BitConverter.ToSingle(b, o); o += 4;
                float z = BitConverter.ToSingle(b, o); o += 4;
                float w = BitConverter.ToSingle(b, o); o += 4;
                data.boneRotations[i] = new Quaternion(x, y, z, w);
            }

            // Hip position (3 floats)
            float hx = BitConverter.ToSingle(b, o); o += 4;
            float hy = BitConverter.ToSingle(b, o); o += 4;
            float hz = BitConverter.ToSingle(b, o); o += 4;
            data.hipPosition = new Vector3(hx, hy, hz);

            // Confidence scores
            data.faceConfidence = BitConverter.ToSingle(b, o); o += 4;
            data.bodyConfidence = BitConverter.ToSingle(b, o); o += 4;

            return data;
        }

        private void EndReceiver()
        {
            if (receiveThread != null)
            {
                stopReception = true;
                receiveThread.Join();
                stopReception = false;
            }
            if (socket != null)
            {
                socket.Close();
                socket = null;
            }
            listening = false;
        }

        void OnApplicationQuit()
        {
            EndReceiver();
        }

        void OnDestroy()
        {
            EndReceiver();
        }
    }
}
