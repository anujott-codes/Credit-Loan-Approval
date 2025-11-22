import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, Shield, Zap, BarChart3, CheckCircle } from 'lucide-react';

const LandingPage = () => {
    return (
        <div className="overflow-hidden">
            {/* Hero Section */}
            <section className="relative bg-white overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-primary-50 to-secondary-50 opacity-50" />
                <div className="container mx-auto px-6 pt-20 pb-32 relative">
                    <div className="flex flex-col lg:flex-row items-center">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.6 }}
                            className="lg:w-1/2 lg:pr-12"
                        >
                            <h1 className="text-5xl lg:text-6xl font-bold text-slate-900 leading-tight mb-6">
                                Financial Approvals, <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-secondary-600">Reimagined.</span>
                            </h1>
                            <p className="text-xl text-slate-600 mb-8 leading-relaxed">
                                Get instant, AI-powered decisions for loans and credit cards.
                                Experience the future of banking with Approv.io's secure and transparent platform.
                            </p>
                            <div className="flex flex-col sm:flex-row gap-4">
                                <Link
                                    to="/loan"
                                    className="inline-flex justify-center items-center px-8 py-4 bg-primary-600 text-white rounded-full font-semibold text-lg hover:bg-primary-700 transition-all shadow-lg hover:shadow-primary-500/30"
                                >
                                    Check Loan Eligibility
                                    <ArrowRight className="ml-2 w-5 h-5" />
                                </Link>
                                <Link
                                    to="/credit"
                                    className="inline-flex justify-center items-center px-8 py-4 bg-white text-slate-700 border border-slate-200 rounded-full font-semibold text-lg hover:bg-slate-50 transition-all"
                                >
                                    Apply for Credit Card
                                </Link>
                            </div>
                            <div className="mt-10 flex items-center space-x-6 text-sm text-slate-500">
                                <div className="flex items-center">
                                    <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                                    <span>No hidden fees</span>
                                </div>
                                <div className="flex items-center">
                                    <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                                    <span>Secure data</span>
                                </div>
                                <div className="flex items-center">
                                    <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                                    <span>Instant results</span>
                                </div>
                            </div>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.8, delay: 0.2 }}
                            className="lg:w-1/2 mt-16 lg:mt-0 relative"
                        >
                            <div className="relative rounded-2xl overflow-hidden shadow-2xl border border-slate-100 bg-white p-2">
                                <img
                                    src="https://images.unsplash.com/photo-1563986768609-322da13575f3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80"
                                    alt="Dashboard Preview"
                                    className="rounded-xl w-full h-auto object-cover"
                                />
                                {/* Floating Cards */}
                                <motion.div
                                    animate={{ y: [0, -10, 0] }}
                                    transition={{ repeat: Infinity, duration: 4 }}
                                    className="absolute -bottom-6 -left-6 bg-white p-4 rounded-xl shadow-xl border border-slate-100"
                                >
                                    <div className="flex items-center space-x-3">
                                        <div className="bg-green-100 p-2 rounded-full">
                                            <CheckCircle className="w-6 h-6 text-green-600" />
                                        </div>
                                        <div>
                                            <p className="text-sm font-medium text-slate-500">Status</p>
                                            <p className="text-lg font-bold text-slate-900">Approved</p>
                                        </div>
                                    </div>
                                </motion.div>
                            </div>
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-24 bg-white">
                <div className="container mx-auto px-6">
                    <div className="text-center max-w-3xl mx-auto mb-16">
                        <h2 className="text-3xl font-bold text-slate-900 mb-4">Why Choose Approv.io?</h2>
                        <p className="text-lg text-slate-600">
                            We leverage cutting-edge machine learning to provide fair, fast, and accurate financial assessments.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                        {[
                            {
                                icon: Zap,
                                title: "Lightning Fast",
                                description: "Get approval decisions in seconds, not days. Our AI processes your application instantly."
                            },
                            {
                                icon: Shield,
                                title: "Bank-Grade Security",
                                description: "Your data is encrypted with 256-bit SSL and never shared with third parties without consent."
                            },
                            {
                                icon: BarChart3,
                                title: "Smart Analytics",
                                description: "Understand your financial health with detailed insights and personalized recommendations."
                            }
                        ].map((feature, index) => (
                            <motion.div
                                key={index}
                                whileHover={{ y: -5 }}
                                className="p-8 rounded-2xl bg-slate-50 border border-slate-100 hover:shadow-lg transition-all"
                            >
                                <div className="w-14 h-14 bg-white rounded-xl shadow-sm flex items-center justify-center mb-6 text-primary-600">
                                    <feature.icon className="w-7 h-7" />
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-3">{feature.title}</h3>
                                <p className="text-slate-600 leading-relaxed">
                                    {feature.description}
                                </p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-24 bg-slate-900 text-white overflow-hidden relative">
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
                <div className="container mx-auto px-6 relative z-10 text-center">
                    <h2 className="text-4xl font-bold mb-6">Ready to get started?</h2>
                    <p className="text-xl text-slate-300 mb-10 max-w-2xl mx-auto">
                        Join thousands of users who have already secured their financial future with Approv.io.
                    </p>
                    <Link
                        to="/loan"
                        className="inline-flex justify-center items-center px-8 py-4 bg-primary-600 text-white rounded-full font-semibold text-lg hover:bg-primary-500 transition-all shadow-lg shadow-primary-900/50"
                    >
                        Apply Now
                        <ArrowRight className="ml-2 w-5 h-5" />
                    </Link>
                </div>
            </section>
        </div>
    );
};

export default LandingPage;
