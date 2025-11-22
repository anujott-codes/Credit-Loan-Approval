import React from 'react';
import { motion } from 'framer-motion';
import { Users, Target, Globe, Award } from 'lucide-react';

const AboutPage = () => {
    return (
        <div className="bg-white">
            {/* Hero Section */}
            <section className="bg-slate-900 text-white py-20">
                <div className="container mx-auto px-6 text-center">
                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-4xl md:text-5xl font-bold mb-6"
                    >
                        About Approv.io
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="text-xl text-slate-300 max-w-2xl mx-auto"
                    >
                        We are on a mission to democratize access to financial services through the power of Artificial Intelligence.
                    </motion.p>
                </div>
            </section>

            {/* Mission & Vision */}
            <section className="py-20">
                <div className="container mx-auto px-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                        <div>
                            <h2 className="text-3xl font-bold text-slate-900 mb-6">Our Mission</h2>
                            <p className="text-lg text-slate-600 leading-relaxed mb-6">
                                To provide instant, transparent, and fair credit and loan approval decisions to everyone, everywhere. We believe that financial opportunities should be accessible to all, not just a privileged few.
                            </p>
                            <div className="grid grid-cols-2 gap-6">
                                <div className="p-4 bg-slate-50 rounded-xl">
                                    <Target className="w-8 h-8 text-primary-600 mb-2" />
                                    <h3 className="font-bold text-slate-900">Accuracy</h3>
                                    <p className="text-sm text-slate-500">99.9% model precision</p>
                                </div>
                                <div className="p-4 bg-slate-50 rounded-xl">
                                    <Globe className="w-8 h-8 text-primary-600 mb-2" />
                                    <h3 className="font-bold text-slate-900">Global</h3>
                                    <p className="text-sm text-slate-500">Serving 50+ countries</p>
                                </div>
                            </div>
                        </div>
                        <div className="relative">
                            <img
                                src="https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80"
                                alt="Team collaboration"
                                className="rounded-2xl shadow-2xl"
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="bg-slate-50 py-20">
                <div className="container mx-auto px-6">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
                        {[
                            { label: "Applications Processed", value: "1M+" },
                            { label: "Partner Banks", value: "50+" },
                            { label: "Countries Served", value: "30+" },
                            { label: "Team Members", value: "100+" }
                        ].map((stat, index) => (
                            <div key={index} className="p-6">
                                <div className="text-4xl font-bold text-primary-600 mb-2">{stat.value}</div>
                                <div className="text-slate-600 font-medium">{stat.label}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Team Section */}
            <section className="py-20">
                <div className="container mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-slate-900 mb-4">Leadership Team</h2>
                        <p className="text-slate-600">Meet the minds behind the revolution.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {[
                            { name: "Sarah Johnson", role: "CEO & Founder", img: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-4.0.3&auto=format&fit=crop&w=687&q=80" },
                            { name: "Michael Chen", role: "CTO", img: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1170&q=80" },
                            { name: "Emily Davis", role: "Head of Product", img: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-4.0.3&auto=format&fit=crop&w=1170&q=80" }
                        ].map((member, index) => (
                            <div key={index} className="text-center group">
                                <div className="relative mb-4 inline-block overflow-hidden rounded-full w-48 h-48">
                                    <img
                                        src={member.img}
                                        alt={member.name}
                                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                                    />
                                </div>
                                <h3 className="text-xl font-bold text-slate-900">{member.name}</h3>
                                <p className="text-primary-600">{member.role}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>
        </div>
    );
};

export default AboutPage;
