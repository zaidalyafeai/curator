'use client'

import { RunsTable } from "@/components/dataset-viewer/RunsTable"
import { useRouter } from "next/navigation"
import Image from "next/image"

export default function Home() {
  const router = useRouter()

  return (
    <div className="container mx-auto p-4">
    
      <div className="flex items-center gap-2 cursor-pointer" onClick={() => router.push('/')}>
        <Image
          src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-sN2O0LK0cVw6NesKNPlJCoWAu7xfOm.png"
          alt="Bespoke Logo"
          width={32}
          height={32}
          className="object-contain"
        />
        <h1 className="text-2xl font-bold mb-4">Bespoke Dataset Completions Dashboard</h1>
      </div>
      <RunsTable />
    </div>
  )
}