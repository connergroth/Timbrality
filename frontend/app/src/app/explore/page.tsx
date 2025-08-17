'use client'

import { useSupabase } from '@/components/SupabaseProvider'
import { useState, useEffect } from 'react'
import { Navbar } from '@/components/Navbar'
import { NavigationSidebar } from '@/components/NavigationSidebar'
import { useSidebar } from '@/contexts/SidebarContext'
import { Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

interface Album {
  id: number
  title: string
  artist: string
  rating: number
  cover: string
  year: number
  genre?: string
}

export default function ExplorePage() {
  const { user, loading, signOut } = useSupabase()
  const { isExpanded } = useSidebar()
  const [searchQuery, setSearchParams] = useState('')
  const [activeTab, setActiveTab] = useState('for-you')
  const [albums, setAlbums] = useState<Album[]>([])

  // Mock data for albums - in production, this would come from your API
  const mockAlbums: Album[] = [
    {
      id: 1,
      title: "God Does Like Ugly",
      artist: "Atmosphere",
      rating: 85,
      cover: "https://i.scdn.co/image/ab67616d0000b273a00b11c129b27a88fc72f36c",
      year: 2002,
      genre: "Hip Hop"
    },
    {
      id: 2,
      title: "Willoughby Tucker: It's Always Love",
      artist: "Soul Coughing",
      rating: 77,
      cover: "https://i.scdn.co/image/ab67616d0000b273c6f158d928315fa0a246028c",
      year: 1996,
      genre: "Alternative"
    },
    {
      id: 3,
      title: "I Love My Computer",
      artist: "Bad Computer",
      rating: 83,
      cover: "https://i.scdn.co/image/ab67616d0000b273f8df5e5b4b9ce7bf1d9e5b71",
      year: 2019,
      genre: "Electronic"
    },
    {
      id: 4,
      title: "BLACK STAR",
      artist: "Yasiin Bey & Talib Kweli",
      rating: 90,
      cover: "https://i.scdn.co/image/ab67616d0000b273b91d2b51bb1c8d3cd1ad9786",
      year: 1998,
      genre: "Hip Hop"
    },
    {
      id: 5,
      title: "Soft Error",
      artist: "Julia Holter",
      rating: 81,
      cover: "https://i.scdn.co/image/ab67616d0000b273c942c3b8b6b6a0b1e8b4a3e2",
      year: 2023,
      genre: "Art Pop"
    },
    {
      id: 6,
      title: "METAL DEATH",
      artist: "Pharmakon",
      rating: 86,
      cover: "https://i.scdn.co/image/ab67616d0000b273d8e5b4c5c7b9e0a1f2d3e4f5",
      year: 2019,
      genre: "Noise"
    },
    {
      id: 7,
      title: "The Last Wax",
      artist: "Boards of Canada",
      rating: 78,
      cover: "https://i.scdn.co/image/ab67616d0000b273e6f7a8b9c0d1e2f3g4h5i6j7",
      year: 2013,
      genre: "IDM"
    },
    {
      id: 8,
      title: "The Sleep No Flowers",
      artist: "Grouper",
      rating: 85,
      cover: "https://i.scdn.co/image/ab67616d0000b273a8b9c0d1e2f3g4h5i6j7k8l9",
      year: 2021,
      genre: "Ambient"
    },
    {
      id: 9,
      title: "DJ Polygon & The Metalheads: Collision",
      artist: "Various Artists",
      rating: 72,
      cover: "https://i.scdn.co/image/ab67616d0000b273b9c0d1e2f3g4h5i6j7k8l9m0",
      year: 2020,
      genre: "Electronic"
    },
    {
      id: 10,
      title: "ABOMINATION REVEALED AT THE EARTH",
      artist: "Death Grips",
      rating: 88,
      cover: "https://i.scdn.co/image/ab67616d0000b273c0d1e2f3g4h5i6j7k8l9m0n1",
      year: 2018,
      genre: "Experimental Hip Hop"
    },
    {
      id: 11,
      title: "The Bitter End: Something Great",
      artist: "Modest Mouse",
      rating: 84,
      cover: "https://i.scdn.co/image/ab67616d0000b273d1e2f3g4h5i6j7k8l9m0n1o2",
      year: 2004,
      genre: "Indie Rock"
    },
    {
      id: 12,
      title: "Blackened III",
      artist: "Deafheaven",
      rating: 91,
      cover: "https://i.scdn.co/image/ab67616d0000b273e2f3g4h5i6j7k8l9m0n1o2p3",
      year: 2021,
      genre: "Blackgaze"
    }
  ]

  useEffect(() => {
    setAlbums(mockAlbums)
  }, [])

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-900 flex items-center justify-center">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-4 border-white/20"></div>
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-white animate-spin"></div>
          <div className="absolute inset-2 rounded-full border-2 border-transparent border-t-white/60 animate-spin" style={{animationDirection: 'reverse', animationDuration: '0.8s'}}></div>
          <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-white rounded-full transform -translate-x-1/2 -translate-y-1/2 animate-pulse"></div>
        </div>
      </div>
    )
  }

  // Show auth page if no user
  if (!user) {
    return (
      <div className="min-h-screen bg-neutral-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-inter font-semibold mb-4 tracking-tight text-white">Please sign in to continue</h1>
          <p className="text-neutral-300 mb-4 font-inter">You need to authenticate to access Timbrality.</p>
          <button 
            onClick={() => window.location.href = '/auth'}
            className="bg-neutral-800 text-white px-6 py-3 rounded-lg font-inter font-medium hover:bg-neutral-700 transition-colors"
          >
            Go to Auth Page
          </button>
        </div>
      </div>
    )
  }

  const filteredAlbums = albums.filter(album =>
    album.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    album.artist.toLowerCase().includes(searchQuery.toLowerCase())
  )


  return (
    <div className="flex min-h-screen bg-neutral-900">
      {/* Navigation Sidebar */}
      <NavigationSidebar 
        user={user}
        onSignOut={signOut}
      />

      {/* Main Content */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${
        isExpanded ? 'ml-40' : 'ml-16'
      }`}>
        <Navbar 
          user={user} 
          onSignOut={signOut}
        />
        
        <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
          {/* Header with Tabs and Search */}
          <div className="flex items-end justify-between mb-8">
            {/* Navigation Tabs */}
            <div className="flex-1">
              <div className="bg-transparent border-b border-neutral-700/50 pb-0 h-auto">
                <div className="flex items-end justify-between">
                  <div className="flex gap-1">
                    <button 
                      onClick={() => setActiveTab('for-you')}
                      className={`px-4 py-2 mb-1 font-medium transition-all duration-200 rounded-t-lg ${
                        activeTab === 'for-you' 
                          ? 'text-white bg-neutral-800 border-b-2 border-white' 
                          : 'text-neutral-400 hover:text-white hover:bg-neutral-800/50'
                      }`}
                    >
                      For You
                    </button>
                    <button 
                      onClick={() => setActiveTab('top-rated')}
                      className={`px-4 py-2 mb-1 font-medium transition-all duration-200 rounded-t-lg ${
                        activeTab === 'top-rated' 
                          ? 'text-white bg-neutral-800 border-b-2 border-white' 
                          : 'text-neutral-400 hover:text-white hover:bg-neutral-800/50'
                      }`}
                    >
                      Top Rated
                    </button>
                    <button 
                      onClick={() => setActiveTab('new')}
                      className={`px-4 py-2 mb-1 font-medium transition-all duration-200 rounded-t-lg ${
                        activeTab === 'new' 
                          ? 'text-white bg-neutral-800 border-b-2 border-white' 
                          : 'text-neutral-400 hover:text-white hover:bg-neutral-800/50'
                      }`}
                    >
                      New
                    </button>
                    <button 
                      onClick={() => setActiveTab('trending')}
                      className={`px-4 py-2 mb-1 font-medium transition-all duration-200 rounded-t-lg ${
                        activeTab === 'trending' 
                          ? 'text-white bg-neutral-800 border-b-2 border-white' 
                          : 'text-neutral-400 hover:text-white hover:bg-neutral-800/50'
                      }`}
                    >
                      Trending
                    </button>
                    <button 
                      onClick={() => setActiveTab('mood')}
                      className={`px-4 py-2 mb-1 font-medium transition-all duration-200 rounded-t-lg ${
                        activeTab === 'mood' 
                          ? 'text-white bg-neutral-800 border-b-2 border-white' 
                          : 'text-neutral-400 hover:text-white hover:bg-neutral-800/50'
                      }`}
                    >
                      By Mood/Tag
                    </button>
                    <button 
                      onClick={() => setActiveTab('decade')}
                      className={`px-4 py-2 mb-1 font-medium transition-all duration-200 rounded-t-lg ${
                        activeTab === 'decade' 
                          ? 'text-white bg-neutral-800 border-b-2 border-white' 
                          : 'text-neutral-400 hover:text-white hover:bg-neutral-800/50'
                      }`}
                    >
                      By Decade/Genre
                    </button>
                  </div>
                  
                  {/* Search Bar */}
                  <div className="pb-1">
                    <Input
                      type="text"
                      placeholder="Search albums or artists..."
                      value={searchQuery}
                      onChange={(e) => setSearchParams(e.target.value)}
                      className="w-80 h-10 rounded-full pl-4 pr-4 bg-neutral-800 border border-neutral-700 focus:ring-0 focus:outline-none focus-visible:ring-0 focus-visible:outline-none focus:ring-offset-0 focus-visible:ring-offset-0 outline-none transition-colors text-sm text-white placeholder:text-neutral-400 focus:border-neutral-600"
                      style={{ outline: 'none', boxShadow: 'none' }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Content Area */}
          <div className="w-full">
            {activeTab === 'for-you' && (
              <div className="mb-6">
                <h2 className="text-xl font-playfair text-white mb-6">Recommended for You</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {filteredAlbums.map((album) => (
                    <div key={album.id} className="group cursor-pointer">
                      <div className="relative mb-3">
                        <img
                          src={album.cover}
                          alt={album.title}
                          className="w-full aspect-square object-cover rounded-lg group-hover:opacity-80 transition-opacity"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement
                            target.src = 'https://via.placeholder.com/300x300/333/fff?text=No+Image'
                          }}
                        />
                        <div className="absolute top-2 right-2 bg-neutral-800/90 text-white text-xs px-2 py-1 rounded-full">
                          {album.rating}
                        </div>
                      </div>
                      <h3 className="font-medium text-white text-sm mb-1 line-clamp-2 group-hover:text-gray-300">
                        {album.title}
                      </h3>
                      <p className="text-muted-foreground text-xs mb-1">
                        {album.artist}
                      </p>
                      <p className="text-muted-foreground/70 text-xs">
                        {album.year}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'top-rated' && (
              <div className="mb-6">
                <h2 className="text-xl font-playfair text-white mb-6">Top Rated Albums</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {filteredAlbums
                    .sort((a, b) => b.rating - a.rating)
                    .map((album) => (
                      <div key={album.id} className="group cursor-pointer">
                        <div className="relative mb-3">
                          <img
                            src={album.cover}
                            alt={album.title}
                            className="w-full aspect-square object-cover rounded-lg group-hover:opacity-80 transition-opacity"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement
                              target.src = 'https://via.placeholder.com/300x300/333/fff?text=No+Image'
                            }}
                          />
                          <div className="absolute top-2 right-2 bg-neutral-800/90 text-white text-xs px-2 py-1 rounded-full">
                            {album.rating}
                          </div>
                        </div>
                        <h3 className="font-medium text-white text-sm mb-1 truncate group-hover:text-neutral-300">
                          {album.title}
                        </h3>
                        <p className="text-neutral-400 text-xs mb-1">
                          {album.artist}
                        </p>
                        <p className="text-neutral-500 text-xs">
                          {album.year}
                        </p>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {activeTab === 'new' && (
              <div className="mb-6">
                <h2 className="text-xl font-playfair text-white mb-6">New Releases</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {filteredAlbums
                    .sort((a, b) => b.year - a.year)
                    .map((album) => (
                      <div key={album.id} className="group cursor-pointer">
                        <div className="relative mb-3">
                          <img
                            src={album.cover}
                            alt={album.title}
                            className="w-full aspect-square object-cover rounded-lg group-hover:opacity-80 transition-opacity"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement
                              target.src = 'https://via.placeholder.com/300x300/333/fff?text=No+Image'
                            }}
                          />
                          <div className="absolute top-2 right-2 bg-neutral-800/90 text-white text-xs px-2 py-1 rounded-full">
                            {album.rating}
                          </div>
                        </div>
                        <h3 className="font-medium text-white text-sm mb-1 truncate group-hover:text-neutral-300">
                          {album.title}
                        </h3>
                        <p className="text-neutral-400 text-xs mb-1">
                          {album.artist}
                        </p>
                        <p className="text-neutral-500 text-xs">
                          {album.year}
                        </p>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {activeTab === 'trending' && (
              <div className="mb-6">
                <h2 className="text-xl font-playfair text-white mb-6">Trending Now</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {filteredAlbums
                    .sort(() => Math.random() - 0.5)
                    .slice(0, 12)
                    .map((album) => (
                      <div key={album.id} className="group cursor-pointer">
                        <div className="relative mb-3">
                          <img
                            src={album.cover}
                            alt={album.title}
                            className="w-full aspect-square object-cover rounded-lg group-hover:opacity-80 transition-opacity"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement
                              target.src = 'https://via.placeholder.com/300x300/333/fff?text=No+Image'
                            }}
                          />
                          <div className="absolute top-2 right-2 bg-neutral-800/90 text-white text-xs px-2 py-1 rounded-full">
                            {album.rating}
                          </div>
                        </div>
                        <h3 className="font-medium text-white text-sm mb-1 truncate group-hover:text-neutral-300">
                          {album.title}
                        </h3>
                        <p className="text-neutral-400 text-xs mb-1">
                          {album.artist}
                        </p>
                        <p className="text-neutral-500 text-xs">
                          {album.year}
                        </p>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {activeTab === 'mood' && (
              <div className="mb-6">
                <h2 className="text-xl font-playfair text-white mb-6">Browse by Mood & Tag</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {filteredAlbums
                    .filter(album => album.genre)
                    .map((album) => (
                      <div key={album.id} className="group cursor-pointer">
                        <div className="relative mb-3">
                          <img
                            src={album.cover}
                            alt={album.title}
                            className="w-full aspect-square object-cover rounded-lg group-hover:opacity-80 transition-opacity"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement
                              target.src = 'https://via.placeholder.com/300x300/333/fff?text=No+Image'
                            }}
                          />
                          <div className="absolute top-2 right-2 bg-neutral-800/90 text-white text-xs px-2 py-1 rounded-full">
                            {album.rating}
                          </div>
                          <div className="absolute bottom-2 left-2 bg-neutral-700/90 text-neutral-200 text-xs px-2 py-1 rounded-full">
                            {album.genre}
                          </div>
                        </div>
                        <h3 className="font-medium text-white text-sm mb-1 truncate group-hover:text-neutral-300">
                          {album.title}
                        </h3>
                        <p className="text-neutral-400 text-xs mb-1">
                          {album.artist}
                        </p>
                        <p className="text-neutral-500 text-xs">
                          {album.year}
                        </p>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {activeTab === 'decade' && (
              <div className="mb-6">
                <h2 className="text-xl font-playfair text-white mb-6">Browse by Decade & Genre</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {filteredAlbums
                    .sort((a, b) => {
                      const decadeA = Math.floor(a.year / 10) * 10
                      const decadeB = Math.floor(b.year / 10) * 10
                      return decadeB - decadeA
                    })
                    .map((album) => (
                      <div key={album.id} className="group cursor-pointer">
                        <div className="relative mb-3">
                          <img
                            src={album.cover}
                            alt={album.title}
                            className="w-full aspect-square object-cover rounded-lg group-hover:opacity-80 transition-opacity"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement
                              target.src = 'https://via.placeholder.com/300x300/333/fff?text=No+Image'
                            }}
                          />
                          <div className="absolute top-2 right-2 bg-neutral-800/90 text-white text-xs px-2 py-1 rounded-full">
                            {album.rating}
                          </div>
                          <div className="absolute bottom-2 left-2 bg-neutral-700/90 text-neutral-200 text-xs px-2 py-1 rounded-full">
                            {Math.floor(album.year / 10) * 10}s
                          </div>
                        </div>
                        <h3 className="font-medium text-white text-sm mb-1 truncate group-hover:text-neutral-300">
                          {album.title}
                        </h3>
                        <p className="text-neutral-400 text-xs mb-1">
                          {album.artist}
                        </p>
                        <p className="text-neutral-500 text-xs">
                          {album.year}
                        </p>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}