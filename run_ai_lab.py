"""Dependency-friendly launcher for the AI LAB web application."""

try:
    from builder.app import app
except SystemExit:
    raise
except Exception as exc:
    raise SystemExit(f"AI LAB could not start: {exc}") from exc

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
