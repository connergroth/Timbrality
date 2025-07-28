import { useState, useEffect } from "react";

export const WhyTimbre = () => {
  const [currentExampleIndex, setCurrentExampleIndex] = useState(0);
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);
  const [displayedMessages, setDisplayedMessages] = useState<{text: string, isUser: boolean}[]>([]);
  const [currentTypingText, setCurrentTypingText] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const demoExamples = [
    [
      { text: "I'm looking for something that feels like a rainy Sunday morning...", isUser: true },
      { text: "Based on your taste for ambient and indie folk, I'd recommend 'Re: Stacks' by Bon Iver. The intimate vocals and sparse instrumentation create that perfect cozy, contemplative atmosphere you're looking for.", isUser: false }
    ],
    [
      { text: "Can you find me songs similar to Radiohead's Kid A but more upbeat?", isUser: true },
      { text: "I found some great matches! Try 'Everything in Its Right Place' energy but with more groove - check out 'Rounds' by Four Tet and 'Since I Left You' by The Avalanches. They have that electronic experimentation with a more danceable pulse.", isUser: false }
    ],
    [
      { text: "Show me music that sounds like the color purple...", isUser: true },
      { text: "Purple has such rich, mysterious tones! I'm thinking 'Teardrop' by Massive Attack for that deep, velvety texture, or 'Purple Rain' by Prince for the regal intensity. Both capture different shades of that color in sound.", isUser: false }
    ]
  ];

  const typewriterSpeed = 30; // milliseconds per character
  const pauseBetweenMessages = 1500; // pause between messages in same example
  const pauseBetweenExamples = 3000; // pause before starting next example

  useEffect(() => {
    const currentExample = demoExamples[currentExampleIndex];
    
    if (currentMessageIndex >= currentExample.length) {
      // Move to next example after pause
      setTimeout(() => {
        setCurrentExampleIndex(prev => (prev + 1) % demoExamples.length);
        setCurrentMessageIndex(0);
        setDisplayedMessages([]);
        setCurrentTypingText("");
      }, pauseBetweenExamples);
      return;
    }

    const currentMessage = currentExample[currentMessageIndex];
    let charIndex = 0;
    setIsTyping(true);
    setCurrentTypingText("");

    const typeInterval = setInterval(() => {
      if (charIndex < currentMessage.text.length) {
        setCurrentTypingText(currentMessage.text.slice(0, charIndex + 1));
        charIndex++;
      } else {
        clearInterval(typeInterval);
        setIsTyping(false);
        
        // Add completed message to displayed messages
        setDisplayedMessages(prev => [...prev, currentMessage]);
        setCurrentTypingText("");
        
        // Move to next message after pause
        setTimeout(() => {
          setCurrentMessageIndex(prev => prev + 1);
        }, pauseBetweenMessages);
      }
    }, typewriterSpeed);

    return () => clearInterval(typeInterval);
  }, [currentExampleIndex, currentMessageIndex]);

  const currentMessage = demoExamples[currentExampleIndex]?.[currentMessageIndex];

  return (
    <section id="agent-demo" className="py-20 scroll-mt-24">
      <div className="max-w-4xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="font-playfair text-4xl md:text-5xl font-bold text-foreground mb-4">
            Meet your AI music curator
          </h2>
          <p className="text-xl text-muted-foreground font-playfair">
            Discover music through natural conversation
          </p>
        </div>

        <div className="bg-card border border-border rounded-2xl p-8 min-h-[200px]">
          <div className="w-full max-w-2xl mx-auto space-y-4">
            {/* Display completed messages */}
            {displayedMessages.map((message, index) => (
              <div key={index} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex max-w-[85%] ${message.isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-2`}>
                  {/* Avatar - Only show for agent */}
                  {!message.isUser && (
                    <div className="w-10 h-10 flex items-center justify-center flex-shrink-0 mr-2 mt-1">
                      <img 
                        src="/soundwhite.png" 
                        alt="Timbre Agent" 
                        className="w-6 h-6 object-contain"
                      />
                    </div>
                  )}

                  {/* Message Content */}
                  <div className={`rounded-lg p-3 ${
                    message.isUser 
                      ? 'bg-muted text-foreground' 
                      : 'bg-muted text-foreground'
                  }`}>
                    <p className="text-sm font-inter whitespace-pre-wrap leading-relaxed">
                      {message.text}
                    </p>
                  </div>
                </div>
              </div>
            ))}

            {/* Display currently typing message */}
            {currentMessage && currentTypingText && (
              <div className={`flex ${currentMessage.isUser ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex max-w-[85%] ${currentMessage.isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-2`}>
                  {/* Avatar - Only show for agent */}
                  {!currentMessage.isUser && (
                    <div className="w-10 h-10 flex items-center justify-center flex-shrink-0 mr-2 mt-1">
                      <img 
                        src="/soundwhite.png" 
                        alt="Timbre Agent" 
                        className="w-6 h-6 object-contain"
                      />
                    </div>
                  )}

                  {/* Message Content */}
                  <div className={`rounded-lg p-3 ${
                    currentMessage.isUser 
                      ? 'bg-muted text-foreground' 
                      : 'bg-muted text-foreground'
                  }`}>
                    <p className="text-sm font-inter whitespace-pre-wrap leading-relaxed">
                      {currentTypingText}
                      <span className={`inline-block w-2 h-5 ml-1 ${isTyping ? 'bg-foreground animate-pulse' : ''}`} style={{ animation: isTyping ? 'blink 1s infinite' : 'none' }} />
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="text-center mt-8">
          <p className="text-muted-foreground text-sm font-inter">
            Try it yourself - describe what you're feeling and let our AI find the perfect soundtrack
          </p>
        </div>
      </div>

      <style jsx>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </section>
  );
};