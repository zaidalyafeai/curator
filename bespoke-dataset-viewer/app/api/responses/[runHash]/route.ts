import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import { homedir } from 'os'
import { join } from 'path'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runHash: string }>  }
): Promise<Response> {  // This is the key change
  try {
    const { runHash } = await params
    const responsesPath = join(homedir(), '.cache', 'curator', runHash, 'responses.jsonl')
    
    const searchParams = request.nextUrl.searchParams
    const lastLineNumber = parseInt(searchParams.get('lastLine') || '0')

    const content = await fs.readFile(responsesPath, 'utf-8')
    const lines = content.split('\n').filter(line => line.trim() !== '')
    
    const newLines = lines.slice(lastLineNumber)
    const jsonData = newLines.map(line => JSON.parse(line))

    return NextResponse.json({
      data: jsonData,
      totalLines: lines.length
    })
  } catch (error) {
    console.error("Error reading responses file:", error)
    return NextResponse.json(
      { error: "Failed to read responses file" },
      { status: 500 }
    )
  }
}
