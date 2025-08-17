'use client'

import { useState, useEffect } from 'react'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { supabase } from '@/lib/supabase'
import { 
  X, 
  User, 
  Settings as SettingsIcon, 
  BarChart3, 
  Database, 
  Calendar,
  Cloud,
  Smartphone,
  HelpCircle,
  Zap,
  RefreshCw,
  LogOut,
  Edit,
  ArrowLeft,
  Code
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface UserProfile {
  id: string
  display_name: string | null
  avatar_url: string | null
  bio: string | null
  location: string | null
  email: string
  created_at: string
}

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  user: any
  onSignOut: () => Promise<void>
}

const sidebarItems = [
  { icon: User, label: 'Account', id: 'account' },
  { icon: BarChart3, label: 'Usage', id: 'usage' },
  { icon: Database, label: 'Data controls', id: 'data' },
]

const devSidebarItems = [
  { icon: Code, label: 'Developer', id: 'developer' },
]

export function SettingsModal({ isOpen, onClose, user, onSignOut }: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState('account')
  const [isInProfileSettings, setIsInProfileSettings] = useState(false)
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [formData, setFormData] = useState({
    display_name: '',
    bio: '',
    location: ''
  })
  const [isSaving, setIsSaving] = useState(false)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [initialDisplayName, setInitialDisplayName] = useState('')
  const [mockMode, setMockMode] = useState(false)
  const [isDeveloper, setIsDeveloper] = useState(false)

  useEffect(() => {
    if (isOpen && user) {
      fetchUserProfile()
      // Check if user is a developer (adjust this logic as needed)
      setIsDeveloper(user.email?.includes('conner') || process.env.NODE_ENV === 'development')
      // Load mock mode status
      const currentMockMode = localStorage.getItem('timbre-mock-ml') === 'true' || 
                             process.env.NEXT_PUBLIC_MOCK_ML === 'true'
      setMockMode(currentMockMode)
    }
  }, [isOpen, user])

  // Update form data when profile changes
  useEffect(() => {
    if (profile) {
      const displayName = profile.display_name || ''
      setFormData({
        display_name: displayName,
        bio: profile.bio || '',
        location: profile.location || ''
      })
      setInitialDisplayName(displayName)
      setHasUnsavedChanges(false)
    }
  }, [profile])

  // Track changes to display name
  useEffect(() => {
    if (formData.display_name !== initialDisplayName) {
      setHasUnsavedChanges(true)
    } else {
      setHasUnsavedChanges(false)
    }
  }, [formData.display_name, initialDisplayName])

  const fetchUserProfile = async () => {
    if (!user) return
    
    try {
      
      // First try to find user by auth ID, then by email if that fails
      let { data, error } = await supabase
        .from('users')
        .select('id, display_name, avatar_url, bio, location, email, created_at')
        .eq('id', user.id)
        .single()
      
      // If not found by ID, try by email (for existing records)
      if (error && error.code === 'PGRST116' && user.email) {
        const { data: emailData, error: emailError } = await supabase
          .from('users')
          .select('id, display_name, avatar_url, bio, location, email, created_at')
          .eq('email', user.email)
          .single()
        
        if (!emailError && emailData) {
          data = emailData
          error = null
        }
      }

      if (error) {
        if (error.code !== 'PGRST116') {
          console.error('Error fetching profile:', error)
        }
        
        // Fallback to user metadata instead of trying to create new record
        setProfile({
          id: user.id,
          display_name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || 'User',
          avatar_url: user.user_metadata?.avatar_url || null,
          bio: null,
          location: null,
          email: user.email || '',
          created_at: user.created_at || new Date().toISOString()
        })
      } else {
        setProfile(data)
      }
    } catch (error) {
      // Fallback to user metadata
      setProfile({
        id: user.id,
        display_name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || 'User',
        avatar_url: user.user_metadata?.avatar_url || null,
        bio: null,
        location: null,
        email: user.email || '',
        created_at: user.created_at || new Date().toISOString()
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleEdit = () => {
    // Reset to current profile data when opening edit mode
    if (profile) {
      setFormData({
        display_name: profile.display_name || '',
        bio: profile.bio || '',
        location: profile.location || ''
      })
      setInitialDisplayName(profile.display_name || '')
      setHasUnsavedChanges(false)
    }
    setIsInProfileSettings(true)
  }

  const handleLogOut = async () => {
    await onSignOut()
    onClose()
  }

  const handleSave = async () => {
    if (!user || !hasUnsavedChanges) return false
    
    setIsSaving(true)
    
    try {
      
      // Try to update existing record by ID first, then by email
      let { error: updateError, count } = await supabase
        .from('users')
        .update({
          display_name: formData.display_name || null,
          updated_at: new Date().toISOString()
        })
        .eq('id', user.id)
        .select('*')
      
      // If no rows updated by ID, try by email
      if (count === 0 && user.email) {
        const { error: emailUpdateError, count: emailCount } = await supabase
          .from('users')
          .update({
            display_name: formData.display_name || null,
            updated_at: new Date().toISOString()
          })
          .eq('email', user.email)
          .select('*')
        
        updateError = emailUpdateError
        count = emailCount
      }

      if (updateError) {
        console.error('Error updating profile:', updateError)
        return false
      }
      
      if (count === 0) {
        // Update the local profile state directly since DB record doesn't exist
        setProfile(prev => prev ? {
          ...prev,
          display_name: formData.display_name || null
        } : null)
      } else {
        await fetchUserProfile()
      }
      
      window.dispatchEvent(new CustomEvent('profileUpdated'))
      setHasUnsavedChanges(false)
      return true
    } catch (error) {
      console.error('Error updating profile:', error)
      return false
    } finally {
      setIsSaving(false)
    }
  }

  const handleClose = async () => {
    if (hasUnsavedChanges) {
      await handleSave()
    }
    onClose()
  }

  const handleBackFromProfile = async () => {
    if (hasUnsavedChanges) {
      await handleSave()
    }
    setIsInProfileSettings(false)
  }

  const renderAccountTab = () => (
    <div className="flex-1 p-6 overflow-y-auto">
      <div className="max-w-2xl">
        {/* User Profile Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src={profile?.avatar_url || user?.user_metadata?.avatar_url || ''} alt={profile?.display_name || ''} />
              <AvatarFallback className="bg-neutral-700 text-white text-xl font-semibold">
                {(profile?.display_name || user?.user_metadata?.full_name || user?.email)?.charAt(0).toUpperCase() || 'U'}
              </AvatarFallback>
            </Avatar>
            <div>
              <h2 className="text-xl font-semibold text-white">
                {profile?.display_name || user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'User'}
              </h2>
              <p className="text-neutral-300 text-sm">{profile?.email || user?.email}</p>
            </div>
          </div>
          <div className="flex space-x-2">
            <Button 
              variant="ghost" 
              size="sm"
              onClick={handleEdit}
              className="bg-neutral-700 hover:bg-neutral-600 text-white hover:text-white h-8 px-3 text-xs rounded-xl border-0"
            >
              <SettingsIcon className="h-3 w-3" />
            </Button>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={handleLogOut}
              className="bg-neutral-700 hover:bg-neutral-600 text-red-400 hover:text-red-300 h-8 px-3 text-xs rounded-xl border-0"
            >
              <LogOut className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Plan Section */}
        <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-playfair text-white">Free</h3>
            <Button variant="secondary" size="sm" className="bg-white text-neutral-900 hover:bg-neutral-100 h-7 px-3 text-xs rounded-md">
              Upgrade
            </Button>
          </div>
          
          {/* Credits Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-neutral-700/30">
              <div className="flex items-center space-x-3">
                <Zap className="h-4 w-4 text-yellow-500" />
                <div>
                  <p className="text-white font-medium text-sm">Credits</p>
                  <p className="text-neutral-300 text-xs">Free credits</p>
                </div>
              </div>
              <span className="text-white font-semibold text-sm">258</span>
            </div>
            
            <div className="flex items-center justify-between py-2">
              <div className="flex items-center space-x-3">
                <RefreshCw className="h-4 w-4 text-neutral-400" />
                <div>
                  <p className="text-white font-medium text-sm">Daily refresh credits</p>
                  <p className="text-neutral-300 text-xs">Refresh to 300 at 00:00 every day</p>
                </div>
              </div>
              <span className="text-white font-semibold text-sm">300</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  const renderProfileSettingsPage = () => (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-neutral-700/30">
        <div className="flex items-center space-x-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleBackFromProfile}
            className="text-white hover:text-white h-8 w-8 p-0"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h1 className="text-lg font-semibold text-white">Profile</h1>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="space-y-6">
          {/* Name Input */}
          <div>
            <label className="block text-sm font-medium text-white mb-2">
              Name
            </label>
            <input
              type="text"
              placeholder="Enter your name"
              value={formData.display_name}
              onChange={(e) => setFormData({ ...formData, display_name: e.target.value })}
              className="w-full bg-neutral-800/40 border border-neutral-700/30 rounded-md px-3 py-2 text-sm text-white placeholder:text-neutral-400 focus:outline-none focus:ring-0 focus:bg-neutral-800/60 focus:border-neutral-600/50"
            />
          </div>

          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-white mb-2">
              Email
            </label>
            <div className="text-neutral-300 text-sm">
              {profile?.email || user?.email}
            </div>
          </div>

          {/* Delete Account */}
          <div className="pt-8">
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-white">Delete account</h3>
              <p className="text-xs text-neutral-300">This will delete your account and all data.</p>
              <Button 
                variant="ghost" 
                className="text-red-400 hover:text-red-300 hover:bg-red-500/10 text-sm p-0 h-auto font-normal"
              >
                Delete account
              </Button>
            </div>
          </div>

          {/* Auto-save indicator */}
          {isSaving && (
            <div className="pt-4 text-center">
              <div className="flex items-center justify-center space-x-2 text-neutral-300 text-sm">
                <div className="w-3 h-3 border-2 border-neutral-600 border-t-white rounded-full animate-spin"></div>
                <span>Saving...</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )

  const renderSettingsTab = () => (
    <div className="flex-1 p-6 overflow-y-auto">
      <div className="max-w-2xl">
        <h2 className="text-xl font-semibold text-white mb-6">Profile Settings</h2>
        
        <div className="space-y-6">
          <p className="text-neutral-300 text-sm">Profile settings are available in the Account tab.</p>
        </div>
      </div>
    </div>
  )

  const renderDeveloperTab = () => (
    <div className="flex-1 p-6 overflow-y-auto">
      <div className="max-w-2xl">
        <h2 className="text-xl font-semibold text-white mb-6">Developer Settings</h2>
        
        <div className="space-y-6">
          {/* Mock Mode Toggle */}
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-base font-medium text-white">Mock Mode</h3>
                <p className="text-neutral-300 text-sm">Enable mock data for testing and development</p>
              </div>
              <button
                onClick={() => {
                  const newMode = !mockMode
                  setMockMode(newMode)
                  
                  if (newMode) {
                    localStorage.setItem('timbre-mock-ml', 'true')
                  } else {
                    localStorage.removeItem('timbre-mock-ml')
                  }
                  
                  // Refresh page to apply changes like the old component
                  setTimeout(() => {
                    window.location.reload()
                  }, 300) // Small delay to show toggle animation
                }}
                className={cn(
                  "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                  mockMode ? "bg-neutral-600" : "bg-neutral-700"
                )}
              >
                <span
                  className={cn(
                    "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                    mockMode ? "translate-x-6" : "translate-x-1"
                  )}
                />
              </button>
            </div>
            <p className="text-neutral-400 text-xs">
              When enabled, the application will use mock data for ML recommendations instead of real API calls. Page will refresh when toggled.
            </p>
          </div>

          {/* Debug Info */}
          <div className="bg-neutral-800/40 backdrop-blur-xl border border-neutral-700/30 rounded-lg p-4">
            <h3 className="text-base font-medium text-white mb-3">Debug Information</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-neutral-300">Environment:</span>
                <span className="text-white font-mono">{process.env.NODE_ENV || 'production'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">User ID:</span>
                <span className="text-white font-mono text-xs">{user?.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">Mock Mode:</span>
                <span className={cn("font-mono", mockMode ? "text-green-400" : "text-neutral-400")}>
                  {mockMode ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  const renderPlaceholderTab = (tabId: string, title: string) => (
    <div className="flex-1 p-6 overflow-y-auto">
      <div className="max-w-2xl">
        <h2 className="text-xl font-semibold text-white mb-4">{title}</h2>
        <p className="text-neutral-300 text-sm">This section is coming soon...</p>
      </div>
    </div>
  )

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-4xl h-[600px] bg-neutral-900 border-neutral-700/30 p-0 overflow-hidden rounded-2xl [&>button]:hidden">
        <div className="flex h-full">
          {/* Sidebar */}
          <div className="w-56 bg-neutral-800 flex flex-col rounded-l-2xl border-r border-neutral-700/30">
            {/* Header */}
            <div className="p-4 flex items-center space-x-1">
              <div className="w-6 h-6 flex items-center justify-center">
                <img 
                  src="/soundwhite.png" 
                  alt="Timbrality" 
                  className="w-5 h-5 object-contain"
                />
              </div>
              <span className="text-lg font-playfair text-base font-semibold text-white">Timbrality</span>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-3 pb-4">
              <ul className="space-y-1">
                {sidebarItems.map((item) => {
                  const Icon = item.icon
                  return (
                    <li key={item.id}>
                      <button
                        onClick={() => {
                          setActiveTab(item.id)
                          setIsInProfileSettings(false)
                        }}
                        className={cn(
                          "w-full flex items-center space-x-3 px-3 py-1.5 rounded-lg text-left transition-colors",
                          activeTab === item.id
                            ? "bg-neutral-700/60 text-white"
                            : "text-neutral-300 hover:text-white hover:bg-neutral-700/40"
                        )}
                      >
                        <Icon className="h-4 w-4" />
                        <span className="text-sm font-medium">{item.label}</span>
                      </button>
                    </li>
                  )
                })}
                {isDeveloper && devSidebarItems.map((item) => {
                  const Icon = item.icon
                  return (
                    <li key={item.id}>
                      <button
                        onClick={() => {
                          setActiveTab(item.id)
                          setIsInProfileSettings(false)
                        }}
                        className={cn(
                          "w-full flex items-center space-x-3 px-3 py-1.5 rounded-lg text-left transition-colors",
                          activeTab === item.id
                            ? "bg-neutral-700/60 text-white"
                            : "text-neutral-300 hover:text-white hover:bg-neutral-700/40"
                        )}
                      >
                        <Icon className="h-4 w-4" />
                        <span className="text-sm font-medium">{item.label}</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </nav>

            {/* Help Button */}
            <div className="px-3 pb-4">
              <button className="w-full flex items-center space-x-3 px-3 py-1.5 text-neutral-300 hover:text-white hover:bg-neutral-700/40 rounded-lg transition-colors">
                <HelpCircle className="h-4 w-4" />
                <span className="text-sm font-medium">Get help</span>
                <div className="ml-auto">
                  <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </div>
              </button>
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 flex flex-col bg-neutral-900 rounded-r-2xl">
            {/* Custom Close Button */}
            <div className="absolute top-4 right-4 z-10">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClose}
                className="text-white hover:text-white h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            
            {isInProfileSettings ? (
              renderProfileSettingsPage()
            ) : (
              <>
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-neutral-700/30">
                  <h1 className="text-lg font-semibold text-white">
                    {[...sidebarItems, ...devSidebarItems].find(item => item.id === activeTab)?.label || 'Account'}
                  </h1>
                </div>

                {/* Content */}
                {isLoading ? (
                  <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-8 h-8 border-2 border-neutral-700 border-t-white rounded-full animate-spin mx-auto mb-4"></div>
                      <p className="text-neutral-300 text-sm">Loading...</p>
                    </div>
                  </div>
                ) : (
                  <>
                    {activeTab === 'account' && renderAccountTab()}
                    {activeTab === 'usage' && renderPlaceholderTab('usage', 'Usage')}
                    {activeTab === 'data' && renderPlaceholderTab('data', 'Data Controls')}
                    {activeTab === 'developer' && renderDeveloperTab()}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}