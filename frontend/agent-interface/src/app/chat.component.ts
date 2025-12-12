import { Component, type OnInit, ViewChild, type ElementRef, inject, ChangeDetectorRef } from "@angular/core"
import { CommonModule } from "@angular/common"
import { FormsModule } from "@angular/forms"
import { DomSanitizer, type SafeHtml } from "@angular/platform-browser"
import { ApiService } from "./api.service"
import { LanguageService } from "./language.service"

export interface ChatMessage {
  id: string
  text: string
  sender: "user" | "assistant"
  timestamp: Date
  isStreaming?: boolean
  sanitizedHtml?: SafeHtml
}

@Component({
  selector: "app-chat",
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: "./chat.component.html",
  styleUrl: "./chat.component.css",
})
export class ChatComponent implements OnInit {
  @ViewChild("messagesContainer") messagesContainer!: ElementRef

  messages: ChatMessage[] = []
  userInput = ""
  isLoading = false

  private apiService = inject(ApiService)
  private languageService = inject(LanguageService)
  private sanitizer = inject(DomSanitizer)
  private cdr = inject(ChangeDetectorRef)

  get isStreamingAssistant(): boolean {
    return this.messages.some((m) => m.isStreaming)
  }

  constructor() {}

  ngOnInit(): void {}

sendMessage(): void {
  if (!this.userInput.trim()) return

  // 1. Add User Message
  const userMessage: ChatMessage = {
    id: Date.now().toString(),
    text: this.userInput,
    sender: "user",
    timestamp: new Date(),
  }
  this.messages.push(userMessage)

  const query = this.userInput
  this.userInput = ""
  this.isLoading = true

  // 2. Add Assistant Message IMMEDIATELY (Empty for now)
  const assistantMessage: ChatMessage = {
    id: (Date.now() + 1).toString(),
    text: "", // Empty initially
    sender: "assistant",
    timestamp: new Date(),
    isStreaming: true, // Mark as streaming
  }
  this.messages.push(assistantMessage) // <--- PUSH NOW
  this.scrollToBottom()

  // 3. Start Stream
  this.apiService.streamPredict(query).subscribe({
    next: (token: string) => {
      assistantMessage.text += token
      assistantMessage.sanitizedHtml = this.sanitizer.bypassSecurityTrustHtml(
        this.formatText(assistantMessage.text)
      )
      this.cdr.detectChanges()
      this.scrollToBottom()
    },
    error: (error: any) => {
      console.error("Streaming error:", error)
      assistantMessage.text = "Error: Could not connect to server."
      assistantMessage.isStreaming = false
      this.isLoading = false
      this.cdr.detectChanges()
      this.scrollToBottom()
    },
    complete: () => {
      assistantMessage.isStreaming = false
      this.isLoading = false
      this.cdr.detectChanges()
      this.scrollToBottom()
    },
  })
}

  clearChat() {
    if (confirm("Hapus riwayat percakapan?")) {
      this.messages = []
    }
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      if (this.messagesContainer) {
        this.messagesContainer.nativeElement.scrollTop = this.messagesContainer.nativeElement.scrollHeight
      }
    }, 0)
  }

  formatText(text: string): string {
    if (!text) return ""
    return text.replace(/\n/g, "<br>").replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
  }
}
