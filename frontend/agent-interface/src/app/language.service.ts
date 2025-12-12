import { Injectable } from "@angular/core"

@Injectable({ providedIn: "root" })
export class LanguageService {
  getTranslation(key: string): string {
    const db: Record<string, string> = {
      error_occurred: "❌ Error: Terjadi masalah saat menghubungkan ke agent.",
    }
    return db[key] || key
  }
}
