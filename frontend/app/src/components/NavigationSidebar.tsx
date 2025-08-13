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
  Settings as SettingsIcon,
  Sparkles,
  BarChart3,
  Search,
  LogOut,
  RefreshCw,
  User as UserIcon,
} from "lucide-react"
import { forceSpotifyReauth, hasRequiredSpotifyScopes } from "@/lib/spotify-auth"

interface NavigationSidebarProps {
  user: User | null
  onSignOut: () => Promise<void>
}

export function NavigationSidebar({ user, onSignOut }: NavigationSidebarProps) {
  if (!user) {
    return null;
  }

  const navigationItems = [
    { icon: Home, href: "/", label: "Home" },
    { icon: Search, href: "/explore", label: "Explore" },
    { icon: MessageSquare, href: "/chat", label: "Chat" },
    { icon: Music, href: "/recommend", label: "Recs" },
    { icon: Sparkles, href: "/insights", label: "Insights" },
    { icon: BarChart3, href: "/analytics", label: "Analytics" },
    { icon: SettingsIcon, href: "/settings", label: "Settings" },
  ]

  const handleSpotifyReauth = async () => {
    try {
      await forceSpotifyReauth()
    } catch (error) {
      console.error("Error re-authenticating Spotify:", error)
    }
  }

  return (
    <aside className="fixed inset-y-0 left-0 z-50 w-16 bg-sidebar border-r border-sidebar-border flex flex-col items-center justify-between py-4">
      {/* Nav icons */}
      <nav className="flex flex-col items-center gap-2">
        {navigationItems.map(({ icon: Icon, href, label }) => (
          <Link key={href} href={href} title={label}>
            <Button
              variant="ghost"
              className="h-10 w-10 p-0 rounded-xl text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground"
            >
              <Icon className="h-5 w-5" />
            </Button>
          </Link>
        ))}
      </nav>

      {/* Avatar dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
              className="h-9 w-9 p-0 rounded-md outline-none focus:outline-none focus-visible:outline-none active:outline-none ring-0 hover:ring-0 focus:ring-0 focus-visible:ring-0 active:ring-0 ring-offset-0 focus-visible:ring-offset-0 hover:bg-sidebar-accent/40 data-[state=open]:bg-sidebar-accent/60"
            title={user?.email || "Profile"}
          >
            <Avatar className="h-7 w-7">
              <AvatarImage src={user?.user_metadata?.avatar_url} alt={user?.email || "User"} />
              <AvatarFallback>
                {user?.email?.charAt(0).toUpperCase() || "U"}
              </AvatarFallback>
            </Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-60">
          <DropdownMenuLabel className="flex items-center gap-2">
            <UserIcon className="h-4 w-4" />
            <div className="flex flex-col">
              <span className="text-sm font-medium">{user.user_metadata?.full_name || user.email}</span>
              {user.email && (
                <span className="text-xs text-muted-foreground truncate max-w-[200px]">{user.email}</span>
              )}
            </div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem asChild>
            <Link href="/profile" className="flex items-center gap-2">
              <UserIcon className="h-4 w-4" />
              <span>Profile</span>
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link href="/settings" className="flex items-center gap-2">
              <SettingsIcon className="h-4 w-4" />
              <span>Settings</span>
            </Link>
          </DropdownMenuItem>
          {!hasRequiredSpotifyScopes(user) && (
            <DropdownMenuItem onClick={handleSpotifyReauth} className="flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              <span>Re-authenticate Spotify</span>
            </DropdownMenuItem>
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={onSignOut} className="flex items-center gap-2">
            <LogOut className="h-4 w-4" />
            <span>Log out</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </aside>
  )
} 