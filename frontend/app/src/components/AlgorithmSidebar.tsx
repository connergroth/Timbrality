'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { X, Brain, BarChart3, Tag, TrendingUp } from 'lucide-react';

interface AlgorithmSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AlgorithmSidebar({ isOpen, onClose }: AlgorithmSidebarProps) {
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />
      
      {/* Sidebar */}
      <div className="fixed right-0 top-0 h-full w-96 bg-background border-l border-border z-50 overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-playfair font-semibold">⚙️ Behind the Algorithm</h2>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* NMF Weights */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                NMF Weights
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm">Atmospheric</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 muted rounded-full h-2">
                    <div className="bg-primary h-2 rounded-full" style={{ width: '85%' }}></div>
                  </div>
                  <span className="text-xs text-muted-foreground">0.85</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Experimental</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 muted rounded-full h-2">
                    <div className="bg-primary h-2 rounded-full" style={{ width: '72%' }}></div>
                  </div>
                  <span className="text-xs text-muted-foreground">0.72</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Lo-fi</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 muted rounded-full h-2">
                    <div className="bg-primary h-2 rounded-full" style={{ width: '68%' }}></div>
                  </div>
                  <span className="text-xs text-muted-foreground">0.68</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* BERT Tag Similarities */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Tag className="h-4 w-4" />
                BERT Tag Similarities
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary" className="text-xs">
                  dreamy (0.92)
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  ambient (0.89)
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  electronic (0.87)
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  atmospheric (0.85)
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  experimental (0.82)
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  lo-fi (0.78)
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* AOTY/Last.fm Influences */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Tag Influences
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="text-sm font-medium mb-2">AOTY Influences</h4>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline" className="text-xs">
                    Post-Rock (High)
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    Shoegaze (Medium)
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    Ambient (High)
                  </Badge>
                </div>
              </div>
              <Separator />
              <div>
                <h4 className="text-sm font-medium mb-2">Last.fm Tags</h4>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline" className="text-xs">
                    electronic (89%)
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    ambient (76%)
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    experimental (72%)
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Algorithm Info */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">How it works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>
                Timbre uses a hybrid approach combining Non-negative Matrix Factorization (NMF) 
                for collaborative filtering and BERT embeddings for semantic understanding.
              </p>
              <p>
                Your music taste is analyzed across multiple dimensions including timbre, 
                mood, and listening patterns to provide personalized recommendations.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}