"use client"

import { User } from "@supabase/supabase-js"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuLabel, DropdownMenuSeparator } from "@/components/ui/dropdown-menu"
import {
  Home,
  MessageSquare,
  Music,
  Sparkles,
  BarChart3,
  Search,
  LogOut,
  User as UserIcon,
  History,
  Settings as SettingsIcon,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react"
import { useState, useEffect } from "react"
import { useRouter, usePathname } from "next/navigation"
import { useSidebar } from "@/contexts/SidebarContext"
import { supabase } from "@/lib/supabase"
import { SettingsModal } from "@/components/SettingsModal"

interface ChatSession {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  messageCount: number
}

interface NavigationSidebarProps {
  user: User | null
  onSignOut: () => Promise<void>
}

export function NavigationSidebar({ user, onSignOut }: NavigationSidebarProps) {
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([])
  const [userProfile, setUserProfile] = useState<any>(null)
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false)
  const { isExpanded, setIsExpanded } = useSidebar()
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    if (user) {
      loadChatHistory()
      loadUserProfile()
      
      // Listen for storage changes to update chat history
      const handleStorageChange = (e: StorageEvent) => {
        if (e.key === `timbre-chats-${user.id}`) {
          loadChatHistory()
        }
      }
      
      // Listen for custom events when chats are updated
      const handleChatUpdate = () => {
        loadChatHistory()
      }
      
      // Listen for profile updates
      const handleProfileUpdate = () => {
        loadUserProfile()
      }
      
      window.addEventListener('storage', handleStorageChange)
      window.addEventListener('chatHistoryUpdated', handleChatUpdate)
      window.addEventListener('profileUpdated', handleProfileUpdate)
      
      return () => {
        window.removeEventListener('storage', handleStorageChange)
        window.removeEventListener('chatHistoryUpdated', handleChatUpdate)
        window.removeEventListener('profileUpdated', handleProfileUpdate)
      }
    }
  }, [user])

  // Auto-close sidebar when navigating to a new page
  useEffect(() => {
    setIsExpanded(false)
  }, [pathname, setIsExpanded])

  const loadChatHistory = () => {
    try {
      const savedSessions = localStorage.getItem(`timbre-chats-${user?.id}`)
      if (savedSessions) {
        const sessions: ChatSession[] = JSON.parse(savedSessions)
        // Sort by timestamp (most recent first) and limit to 8
        const sortedSessions = sessions
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
          .slice(0, 8)
        setChatHistory(sortedSessions)
      }
    } catch (error) {
      console.error('Error loading chat history:', error)
    }
  }

  const loadUserProfile = async () => {
    if (!user) return
    
    // Try to load from cache first
    const cachedProfile = localStorage.getItem(`user-profile-${user.id}`)
    if (cachedProfile) {
      try {
        const parsedProfile = JSON.parse(cachedProfile)
        setUserProfile(parsedProfile)
      } catch (error) {
        console.error('Error parsing cached profile in NavigationSidebar:', error)
      }
    }
    
    try {
      
      // Try to get display_name from database first by ID, then by email
      let { data, error } = await supabase
        .from('users')
        .select('display_name')
        .eq('id', user.id)
        .single()
      
      // If not found by ID, try by email (for existing records)
      if (error && error.code === 'PGRST116' && user.email) {
        const { data: emailData, error: emailError } = await supabase
          .from('users')
          .select('display_name')
          .eq('email', user.email)
          .single()
        
        if (!emailError && emailData) {
          data = emailData
          error = null
        }
      }
      
      if (error) {
        
        // Always fallback to user metadata instead of trying to create records
        const fallbackProfile = { display_name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || null }
        setUserProfile(fallbackProfile)
        // Cache the fallback profile
        localStorage.setItem(`user-profile-${user.id}`, JSON.stringify(fallbackProfile))
      } else {
        setUserProfile(data)
        // Cache the profile
        localStorage.setItem(`user-profile-${user.id}`, JSON.stringify(data))
      }
    } catch (error) {
      console.error('Exception loading user profile:', error)
      // Fallback to user metadata if there's an exception
      const fallbackProfile = { display_name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || null }
      setUserProfile(fallbackProfile)
      // Cache the fallback profile
      localStorage.setItem(`user-profile-${user.id}`, JSON.stringify(fallbackProfile))
    }
  }

  if (!user) {
    return null;
  }

  const navigationItems = [
    { icon: Home, href: "/", label: "Home" },
    { icon: Search, href: "/explore", label: "Explore" },
    { icon: Music, href: "/recommend", label: "Recs" },
    { icon: Sparkles, href: "/insights", label: "Insights" },
    { icon: BarChart3, href: "/analytics", label: "Analytics" },
  ]

  return (
    <aside className={`fixed inset-y-0 left-0 z-50 bg-sidebar border-r border-sidebar-border flex flex-col py-4 transition-all duration-300 ease-in-out ${
      isExpanded ? 'w-40' : 'w-16'
    }`}>
      {/* Nav icons */}
      <nav className={`flex flex-col gap-2 ${isExpanded ? 'w-full px-3' : 'items-center'}`}>
        {/* Toggle Button and Navigation Header Row */}
        <div className={`flex items-center gap-2 mb-2 ${isExpanded ? 'w-full' : 'justify-center'}`}>
          <Button
            variant="ghost"
            className="p-0 rounded-xl text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground h-8 w-8"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? "Collapse Sidebar" : "Expand Sidebar"}
          >
            {isExpanded ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeft className="h-4 w-4" />}
          </Button>
        </div>


        {navigationItems.map(({ icon: Icon, href, label }) => (
          <Link key={href} href={href} className={isExpanded ? 'w-full' : ''}>
            <Button
              variant="ghost"
              className={`p-0 rounded-xl text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground ${
                isExpanded ? 'w-full h-7 justify-start px-2' : 'h-8 w-8'
              }`}
            >
              <Icon className="h-5 w-5" />
              {isExpanded && <span className="ml-1 text-xs font-medium">{label}</span>}
            </Button>
          </Link>
        ))}
      </nav>

      {/* Chat History */}
      {isExpanded && (
        <div className="flex flex-col gap-2 w-full px-3">
          <div className="w-full mb-1 mt-6">
            <h3 className="text-[11px] font-medium text-sidebar-foreground/60 uppercase tracking-wider pl-2">
              Chats
            </h3>
          </div>
          
          {chatHistory.map((chat) => (
            <Link key={chat.id} href={`/chat/${chat.id}`} className="w-full">
              <Button
                variant="ghost"
                className="p-0 rounded-lg text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground w-full h-6 justify-start px-2"
                title={chat.title || 'Untitled Chat'}
              >
                <span className="text-xs truncate max-w-[140px]">
                  {chat.title || 'Untitled Chat'}
                </span>
              </Button>
            </Link>
          ))}
          
          {/* History Icon */}
          <Link href="/chat" className="w-full">
            <Button
              variant="ghost"
              className="p-0 rounded-xl text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground w-full h-7 justify-start px-2"
              title="Chat History"
            >
              <History className="h-5 w-5" />
              <span className="ml-1 text-xs font-medium">History</span>
            </Button>
          </Link>
        </div>
      )}

      {/* Spacer to push avatar to bottom */}
      <div className="flex-1"></div>

      {/* Avatar dropdown */}
      <div className={`${isExpanded ? 'px-3' : 'flex justify-center'}`}>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              className={`p-0 rounded-md outline-none focus:outline-none focus-visible:outline-none active:outline-none ring-0 hover:ring-0 focus:ring-0 focus-visible:ring-0 active:ring-0 ring-offset-0 focus-visible:ring-offset-0 hover:bg-sidebar-accent/40 data-[state=open]:bg-sidebar-accent/60 ${
                isExpanded ? 'w-full h-7 justify-start px-2' : 'h-7 w-7'
              }`}
              title={user?.email || "Profile"}
            >
              <Avatar className="h-6 w-6">
                <AvatarImage src={user?.user_metadata?.avatar_url} alt={user?.email || "User"} />
                <AvatarFallback>
                  {user?.email?.charAt(0).toUpperCase() || "U"}
                </AvatarFallback>
              </Avatar>
              {isExpanded && (
                <span className="ml-2 text-sm font-medium text-sidebar-foreground truncate">
                  {userProfile?.display_name || user?.email?.split('@')[0] || 'User'}
                </span>
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-60 bg-sidebar border-sidebar-border">
            <DropdownMenuLabel className="flex items-center gap-2 text-sidebar-foreground">
              <UserIcon className="h-4 w-4" />
              <div className="flex flex-col">
                <span className="text-sm font-medium">{userProfile?.display_name || user?.email}</span>
                {user?.email && (
                  <span className="text-xs text-sidebar-foreground/70 truncate max-w-[200px]">{user?.email}</span>
                )}
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => setIsSettingsModalOpen(true)}
              className="flex items-center gap-2 cursor-pointer text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
            >
              <SettingsIcon className="h-4 w-4" />
              <span>Settings</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={onSignOut} className="flex items-center gap-2 text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground">
              <LogOut className="h-4 w-4" />
              <span>Log out</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Settings Modal */}
      <SettingsModal 
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        user={user}
        onSignOut={onSignOut}
      />
    </aside>
  )
} 