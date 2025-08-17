'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  X, 
  BrainCircuit, 
  BarChart3, 
  Tag, 
  TrendingUp, 
  Zap,
  Target,
  Activity,
  Info
} from 'lucide-react';

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
      <div className="fixed right-0 top-0 h-full w-96 bg-neutral-800 border-l border-neutral-700 z-50 overflow-y-auto shadow-2xl">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-8 pb-6 bg-neutral-700/40 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-neutral-600/30 rounded-lg">
                <BrainCircuit className="h-5 w-5 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-inter font-semibold text-white">Algorithm Insights</h2>
                <p className="text-sm text-neutral-300">Behind the music magic</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 hover:bg-neutral-700/60 text-white"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Content Sections */}
          <div className="space-y-6">
            {/* NMF Weights */}
            <Card className="border-neutral-600 bg-neutral-700/30">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <div className="p-1.5 bg-neutral-600/30 rounded">
                    <BarChart3 className="h-4 w-4 text-white" />
                  </div>
                  NMF Weights
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-white">Atmospheric</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-neutral-600 rounded-full h-2">
                        <div className="bg-white h-2 rounded-full transition-all" style={{ width: '85%' }}></div>
                      </div>
                      <span className="text-xs text-neutral-300 font-mono">0.85</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-white">Experimental</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-neutral-600 rounded-full h-2">
                        <div className="bg-white h-2 rounded-full transition-all" style={{ width: '72%' }}></div>
                      </div>
                      <span className="text-xs text-neutral-300 font-mono">0.72</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-white">Lo-fi</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-neutral-600 rounded-full h-2">
                        <div className="bg-white h-2 rounded-full transition-all" style={{ width: '68%' }}></div>
                      </div>
                      <span className="text-xs text-neutral-300 font-mono">0.68</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* BERT Tag Similarities */}
            <Card className="border-neutral-600 bg-neutral-700/30">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <div className="p-1.5 bg-neutral-600/30 rounded">
                    <Tag className="h-4 w-4 text-white" />
                  </div>
                  BERT Tag Similarities
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary" className="text-xs bg-neutral-600/20 text-white border-neutral-600/30">
                    dreamy (0.92)
                  </Badge>
                  <Badge variant="secondary" className="text-xs bg-neutral-600/20 text-white border-neutral-600/30">
                    ambient (0.89)
                  </Badge>
                  <Badge variant="secondary" className="text-xs bg-neutral-600/20 text-white border-neutral-600/30">
                    electronic (0.87)
                  </Badge>
                  <Badge variant="secondary" className="text-xs bg-neutral-600/20 text-white border-neutral-600/30">
                    atmospheric (0.85)
                  </Badge>
                  <Badge variant="secondary" className="text-xs bg-neutral-600/20 text-white border-neutral-600/30">
                    experimental (0.82)
                  </Badge>
                  <Badge variant="secondary" className="text-xs bg-neutral-600/20 text-white border-neutral-600/30">
                    lo-fi (0.78)
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Tag Influences */}
            <Card className="border-neutral-600 bg-neutral-700/30">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <div className="p-1.5 bg-neutral-600/30 rounded">
                    <TrendingUp className="h-4 w-4 text-white" />
                  </div>
                  Tag Influences
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium mb-2 text-white flex items-center gap-2">
                    <Target className="h-3 w-3 text-white" />
                    AOTY Influences
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline" className="text-xs border-neutral-600/40 text-neutral-300">
                      Post-Rock (High)
                    </Badge>
                    <Badge variant="outline" className="text-xs border-neutral-600/40 text-neutral-300">
                      Shoegaze (Medium)
                    </Badge>
                    <Badge variant="outline" className="text-xs border-neutral-600/40 text-neutral-300">
                      Ambient (High)
                    </Badge>
                  </div>
                </div>
                <div className="pt-2">
                  <h4 className="text-sm font-medium mb-2 text-white flex items-center gap-2">
                    <Activity className="h-3 w-3 text-white" />
                    Last.fm Tags
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline" className="text-xs border-neutral-600/40 text-neutral-300">
                      electronic (89%)
                    </Badge>
                    <Badge variant="outline" className="text-xs border-neutral-600/40 text-neutral-300">
                      ambient (76%)
                    </Badge>
                    <Badge variant="outline" className="text-xs border-neutral-600/40 text-neutral-300">
                      experimental (72%)
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Algorithm Info */}
            <Card className="border-neutral-600 bg-neutral-700/30">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <div className="p-1.5 bg-neutral-600/30 rounded">
                    <Info className="h-4 w-4 text-white" />
                  </div>
                  How it works
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-neutral-300 leading-relaxed">
                  Timbrality uses a hybrid approach combining <strong>Non-negative Matrix Factorization (NMF)</strong> 
                  for collaborative filtering and <strong>BERT embeddings</strong> for semantic understanding.
                </p>
                <p className="text-sm text-neutral-300 leading-relaxed">
                  Your music taste is analyzed across multiple dimensions including timbre, 
                  mood, and listening patterns to provide personalized recommendations.
                </p>
                <div className="pt-2 flex items-center gap-2 text-xs text-neutral-400">
                  <Zap className="h-3 w-3" />
                  Powered by advanced ML algorithms
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </>
  );
}