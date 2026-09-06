import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

if (import.meta.env.DEV && import.meta.env.VITE_DISABLE_REACT_DEVTOOLS !== "1") {
  void Promise.all([import("react-grab"), import("react-scan")])
}

const root = document.getElementById("root")

if (root === null) {
  throw new Error("The application root is missing.")
}

createRoot(root).render(<StrictMode />)
