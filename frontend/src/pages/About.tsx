import React from 'react';
import { Target, Users, Zap, Heart, Award, Lightbulb, Shield } from 'lucide-react';

const About: React.FC = () => {
  const technologies = [
    { name: 'React', icon: '⚛️', description: 'Modern UI framework', color: 'blue' },
    { name: 'TypeScript', icon: '🔷', description: 'Type-safe development', color: 'blue' },
    { name: 'FastAPI', icon: '🚀', description: 'High-performance Python API', color: 'green' },
    { name: 'PyTorch', icon: '🔥', description: 'AI/ML framework', color: 'orange' },
    { name: 'Three.js', icon: '🎮', description: '3D graphics and visualization', color: 'yellow' },
    { name: 'Blender', icon: '🎨', description: '3D modeling and animation', color: 'orange' },
    { name: 'LangChain', icon: '🔗', description: 'LLM application framework', color: 'purple' },
    { name: 'MediaPipe', icon: '🤖', description: 'ML gesture recognition', color: 'green' },
    { name: 'OpenCV', icon: '👁️', description: 'Computer vision library', color: 'purple' },
    { name: 'Groq LLM', icon: '🧠', description: 'Advanced language processing', color: 'cyan' },
    { name: 'SQLite', icon: '💾', description: 'Lightweight database', color: 'gray' },
    { name: 'SQLAlchemy', icon: '🗄️', description: 'Database ORM', color: 'indigo' },
    { name: 'Ngrok', icon: '🌐', description: 'Secure tunneling service', color: 'blue' },
    { name: 'Vercel', icon: '▲', description: 'Cloud deployment platform', color: 'black' },
    { name: 'WebSocket', icon: '⚡', description: 'Real-time communication', color: 'red' },
  ];

  const creator = {
    name: 'Zahid Mittha',
    role: 'Creator of ASLense',
    avatar: '👨‍💻',
    color: 'blue',
    cgpa: '3.85/4',
    bio: 'Passionate AI and software engineer with a CGPA of 3.85/4, known for combining innovation with social impact. He is the creator of ASLense, an intelligent sign language interpretation system for the deaf and mute community. Zahid has represented his university at multiple national and regional competitions, including the prestigious NUMS Competition, and is a former university-level cricketer. With strong experience in machine learning, computer vision, NLP, and full-stack development, Zahid continues to develop inclusive solutions that make a real difference in people\'s lives.',
    achievements: [
      '🏆 Represented university at prestigious NUMS Competition',
      '🤖 Expert in Machine Learning & Computer Vision',
      '💻 Full-stack developer with NLP expertise',
      '🌟 Multiple national & regional competition participant'
    ],
    expertise: ['Machine Learning', 'Computer Vision', 'NLP', 'Full-Stack Development', 'PyTorch', 'React', 'FastAPI', 'AI Research']
  };

  const values = [
    {
      icon: Shield,
      title: 'Accessibility',
      description: 'We believe technology should be inclusive and accessible to everyone, regardless of their abilities.',
      color: 'blue'
    },
    {
      icon: Lightbulb,
      title: 'Innovation',
      description: 'We push the boundaries with dual AI models, agentic LLM pipelines, and real-time processing for the best ASL recognition experience.',
      color: 'yellow'
    },
    {
      icon: Users,
      title: 'Community',
      description: 'Built with the deaf community in mind, featuring user authentication, progress tracking, and admin dashboards for comprehensive learning management.',
      color: 'green'
    }
  ];

  const stats = [
    { number: '2200+', label: 'ASL Signs Supported', icon: '🤟', color: 'blue' },
    { number: '5 Models', label: 'AI Processing Options', icon: '🧠', color: 'purple' },
    { number: '95%+', label: 'Recognition Accuracy', icon: '🎯', color: 'green' },
    { number: 'Real-time & sentence-level', label: 'Live Processing', icon: '⚡', color: 'red' },
  ];

  const floatingElements = [
    { emoji: '🤟', delay: 0, duration: 4 },
    { emoji: '👋', delay: 1, duration: 5 },
    { emoji: '✋', delay: 2, duration: 3 },
    { emoji: '👌', delay: 3, duration: 6 },
    { emoji: '🤲', delay: 4, duration: 4 },
    { emoji: '👐', delay: 5, duration: 5 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mission Section */}
      <section className="relative bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 py-20 overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-20 left-20 w-32 h-32 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
          <div className="absolute bottom-20 right-20 w-32 h-32 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
          <div className="absolute top-40 right-40 w-32 h-32 bg-pink-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
        </div>

        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="animate-fade-in-up">
            <div className="relative group mb-8">
              <div className="absolute -inset-4 blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Target className="relative h-16 w-16 text-blue-600 mx-auto animate-pulse" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">Our Mission</h1>
            <p className="text-xl text-gray-600 leading-relaxed">
              ASLense is an AI-powered American Sign Language learning and translation platform that combines 
              computer vision, machine learning, and modern web technologies. We provide interactive, 
              real-time sign language recognition with both mini (fast) and pro (accurate) AI models, 
              creating an inclusive communication bridge for the deaf and hearing communities.
            </p>
             {floatingElements.map((element, i) => (
            <div
              key={i}
              className="absolute text-4xl opacity-20 animate-float pointer-events-none"
              style={{
                left: `${10 + (i * 15)}%`,
                top: `${20 + (i * 10)}%`,
                animationDelay: `${element.delay}s`,
                animationDuration: `${element.duration}s`
              }}
            >
              {element.emoji}
            </div>
          ))}
           {/* Rotating rings */}
                      <div className="absolute inset-0 border-2 border-blue-400/30 rounded-full animate-spin" style={{ animationDuration: '20s' }}></div>
                      <div className="absolute inset-4 border-2 border-purple-400/30 rounded-full animate-spin" style={{ animationDuration: '15s', animationDirection: 'reverse' }}></div>
                      <div className="absolute inset-8 border-2 border-pink-400/30 rounded-full animate-spin" style={{ animationDuration: '10s' }}></div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-white">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-xl text-gray-600">
              Advanced AI technology meets intuitive design
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { emoji: '📹', title: 'Capture', description: 'Upload videos or use live camera to capture ASL gestures with our real-time WebSocket processing.', color: 'blue' },
              { emoji: '🧠', title: 'AI Analysis', description: 'Choose between Mini (fast) or Pro (accurate) models. Pro uses agentic pipeline with Groq LLM for enhanced sentence generation.', color: 'purple' },
              { emoji: '💬', title: 'Translate', description: 'Get real-time word and sentence predictions with confidence scores, plus detailed analytics and progress tracking.', color: 'green' }
            ].map((step, index) => (
              <div
                key={index}
                className={`group text-center p-8 rounded-2xl bg-gradient-to-br from-${step.color}-50 to-${step.color}-100 hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                style={{ animationDelay: `${index * 200}ms` }}
              >
                <div className={`bg-${step.color}-100 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <span className="text-3xl">{step.emoji}</span>
                </div>
                <h3 className={`text-xl font-semibold text-${step.color}-900 mb-4 group-hover:text-${step.color}-700 transition-colors duration-300`}>
                  {step.title}
                </h3>
                <p className={`text-${step.color}-700 leading-relaxed`}>
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technologies */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Zap className="relative h-12 w-12 text-yellow-500 mx-auto animate-bounce" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Powered by Modern Technology</h2>
            <p className="text-xl text-gray-600">
              Built with cutting-edge tools and frameworks
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
            {technologies.map((tech, index) => (
              <div
                key={index}
                className={`group bg-white rounded-xl shadow-sm p-6 hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center space-x-4">
                  <div className={`text-4xl p-3 bg-${tech.color}-50 rounded-xl group-hover:scale-110 transition-transform duration-300`}>
                    {tech.icon}
                  </div>
                  <div>
                    <h3 className={`text-lg font-semibold text-gray-900 group-hover:text-${tech.color}-600 transition-colors duration-300`}>
                      {tech.name}
                    </h3>
                    <p className="text-gray-600">{tech.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Creator */}
      <section className="py-20 bg-gradient-to-br from-indigo-50 via-white to-cyan-50">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Users className="relative h-12 w-12 text-blue-500 mx-auto animate-bounce" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Meet the Creator</h2>
            <p className="text-xl text-gray-600">
              The visionary behind ASLense
            </p>
          </div>

          <div className="flex justify-center">
            <div className="max-w-2xl w-full">
              <div className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up border border-gray-100">
                <div className="text-center mb-6">
                  <div className="relative mx-auto mb-6">
                    <div className={`absolute -inset-2 blur opacity-25 group-hover:opacity-40 transition duration-1000`}></div>
                    <div className="relative w-32 h-32 rounded-full mx-auto bg-gradient-to-r from-blue-400 to-purple-600 flex items-center justify-center border-4 border-white shadow-lg">
                      <span className="text-6xl">{creator.avatar}</span>
                    </div>
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">{creator.name}</h3>
                  <p className={`text-lg font-semibold bg-gradient-to-r from-${creator.color}-500 to-purple-600 bg-clip-text text-transparent mb-2`}>
                    {creator.role}
                  </p>
                  <p className="text-sm text-gray-500 font-medium">CGPA: {creator.cgpa}</p>
                </div>

                <div className="space-y-6">
                  <p className="text-gray-600 leading-relaxed text-center">
                    {creator.bio}
                  </p>

                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3 text-center">🏆 Key Achievements</h4>
                    <div className="space-y-2">
                      {creator.achievements.map((achievement, idx) => (
                        <div key={idx} className="flex items-center justify-center text-sm text-gray-600">
                          <span className="text-center">{achievement}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3 text-center">💻 Technical Expertise</h4>
                    <div className="flex flex-wrap justify-center gap-2">
                      {creator.expertise.map((skill, idx) => (
                        <span
                          key={idx}
                          className={`px-3 py-1 text-xs font-medium rounded-full bg-gradient-to-r from-${creator.color}-500 to-purple-600 text-white shadow-sm`}
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-20 bg-gradient-to-br from-purple-50 to-pink-50">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Heart className="relative h-16 w-16 text-red-500 mx-auto animate-pulse" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Our Values</h2>
            <p className="text-xl text-gray-600">
              The principles that guide everything we do
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {values.map((value, index) => {
              const Icon = value.icon;
              return (
                <div
                  key={index}
                  className={`group text-center p-8 bg-white rounded-2xl shadow-sm hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                  style={{ animationDelay: `${index * 200}ms` }}
                >
                  <div className={`bg-${value.color}-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className={`h-8 w-8 text-${value.color}-600`} />
                  </div>
                  <h3 className={`text-xl font-semibold text-gray-900 mb-4 group-hover:text-${value.color}-600 transition-colors duration-300`}>
                    {value.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {value.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Statistics */}
      <section className="py-20 bg-white">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Award className="relative h-12 w-12 text-green-600 mx-auto animate-bounce" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Making a Difference</h2>
            <p className="text-xl text-gray-600">
              Numbers that showcase our impact
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div
                key={index}
                className={`group text-center p-8 bg-gradient-to-br from-${stat.color}-50 to-${stat.color}-100 rounded-2xl shadow-sm hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                style={{ animationDelay: `${index * 150}ms` }}
              >
                <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  {stat.icon}
                </div>
                <div className={`text-4xl font-bold text-${stat.color}-600 mb-2 animate-counter`}>
                  {stat.number}
                </div>
                <div className={`text-${stat.color}-700 font-medium`}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;