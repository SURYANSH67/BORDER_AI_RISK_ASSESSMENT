import os
import sys
import http.server
import socketserver
import webbrowser

PORT = 8000

def start_dashboard_server():
    """
    Starts a local HTTP server for the GIS Border Security Intelligence Dashboard.
    """
    web_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(web_dir)

    Handler = http.server.SimpleHTTPRequestHandler
    
    print("=" * 70)
    print(f"🛡️  GIS BORDER SECURITY INTELLIGENCE DASHBOARD SERVER")
    print("=" * 70)
    print(f"Server running at: http://localhost:{PORT}/dashboard/index.html")
    print("Press Ctrl+C to stop the server.")
    print("=" * 70)

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard Server] Stopped.")

if __name__ == "__main__":
    start_dashboard_server()
