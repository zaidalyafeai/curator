"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Copy, X } from "lucide-react"
import { DataItem } from "@/types/dataset"
import { useCallback } from "react"

interface DetailsSidebarProps {
  item: DataItem | null
  onClose: () => void
}

export function DetailsSidebar({ item, onClose }: DetailsSidebarProps) {
  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      alert("Copied to clipboard!")
    } catch (err) {
      console.error("Failed to copy:", err)
      alert("Failed to copy to clipboard")
    }
  }, [])

  if (!item) return null

  const [requestData, responseData] = item

  return (
    <Card className="fixed right-0 top-0 h-full w-1/3 rounded-none border-l translate-y-[64px] h-[calc(100vh-64px)]">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-2xl font-bold">Details</CardTitle>
        <Button variant="ghost" size="icon" onClick={onClose} className="rounded-full">
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </Button>
      </CardHeader>
      <ScrollArea className="h-[calc(100vh-10rem)] px-6">
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">Model</h3>
            <p className="text-sm text-muted-foreground">{requestData.model}</p>
          </div>
          <Separator />
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">User Message</h3>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">
              {requestData.messages.find(m => m.role === "user")?.content}
            </p>
            <Button 
              onClick={() => copyToClipboard(requestData.messages.find(m => m.role === "user")?.content || "")} 
              variant="outline" 
              size="sm" 
              className="mt-2"
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
          <Separator />
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">Assistant Message</h3>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">
              {responseData.choices[0]?.message?.content}
            </p>
            <Button 
              onClick={() => copyToClipboard(responseData.choices[0]?.message?.content || "")} 
              variant="outline" 
              size="sm" 
              className="mt-2"
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
          <Separator />
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">Token Usage</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-1">
                <p className="text-sm font-medium">Total</p>
                <p className="text-2xl font-bold">{responseData.usage.total_tokens}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Prompt</p>
                <p className="text-2xl font-bold">{responseData.usage.prompt_tokens}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Completion</p>
                <p className="text-2xl font-bold">{responseData.usage.completion_tokens}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </ScrollArea>
    </Card>
  )
}