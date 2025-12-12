import { Injectable } from "@angular/core"
import { Observable } from "rxjs"

@Injectable({ providedIn: "root" })
export class ApiService {
  streamPredict(query: string): Observable<string> {
    return new Observable<string>((observer) => {
      const controller = new AbortController()

      const apiUrl = this.getApiUrl()

      fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream", // Explicit accept header
        },
        body: JSON.stringify({ message: query }),
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`)
          }
          if (!response.body) {
            throw new Error("No response body received")
          }
          const reader = response.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ""

          while (true) {
            const { done, value } = await reader.read()
            if (done) break
          
            buffer += decoder.decode(value, { stream: true })
          
            // Split by SSE event boundaries (double newline)
            const lines = buffer.split("\n\n")
            // Keep the last incomplete line in the buffer
            buffer = lines.pop() || ""
          
            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const data = line.substring(6)
              
                // Skip [DONE] marker
                if (data === "[DONE]") continue
              
                if (data) {
                  observer.next(data)
                }
              }
            }
          }
          observer.complete()
        })
        .catch((err) => {
          console.error("[v0] Fetch error details:", err)
          observer.error(err)
        })

      return () => controller.abort()
    })
  }

  private getApiUrl(): string {
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
      return "http://127.0.0.1:5000"
    }
    return "" // Use same origin in production
  }
}
