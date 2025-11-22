import React from 'react';
import { CreditCard, Twitter, Linkedin, Github, Mail } from 'lucide-react';

const Footer = () => {
    return (
        <footer className="bg-slate-900 text-slate-300 py-12 border-t border-slate-800">
            <div className="container mx-auto px-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
                    <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                            <CreditCard className="text-primary-400 w-6 h-6" />
                            <span className="text-2xl font-bold text-white">Approv.io</span>
                        </div>
                        <p className="text-sm text-slate-400">
                            Empowering your financial future with instant, AI-driven credit and loan approvals.
                        </p>
                    </div>

                    <div>
                        <h3 className="text-white font-semibold mb-4">Product</h3>
                        <ul className="space-y-2 text-sm">
                            <li><a href="/loan" className="hover:text-primary-400 transition-colors">Loan Approval</a></li>
                            <li><a href="/credit" className="hover:text-primary-400 transition-colors">Credit Cards</a></li>
                            <li><a href="/about" className="hover:text-primary-400 transition-colors">About Us</a></li>
                        </ul>
                    </div>

                    <div>
                        <h3 className="text-white font-semibold mb-4">Legal</h3>
                        <ul className="space-y-2 text-sm">
                            <li><a href="#" className="hover:text-primary-400 transition-colors">Privacy Policy</a></li>
                            <li><a href="#" className="hover:text-primary-400 transition-colors">Terms of Service</a></li>
                            <li><a href="#" className="hover:text-primary-400 transition-colors">Cookie Policy</a></li>
                        </ul>
                    </div>

                    <div>
                        <h3 className="text-white font-semibold mb-4">Contact</h3>
                        <ul className="space-y-2 text-sm">
                            <li className="flex items-center space-x-2">
                                <Mail className="w-4 h-4" />
                                <span>support@approv.io</span>
                            </li>
                            <li className="flex space-x-4 mt-4">
                                <a href="#" className="hover:text-white transition-colors"><Twitter className="w-5 h-5" /></a>
                                <a href="#" className="hover:text-white transition-colors"><Linkedin className="w-5 h-5" /></a>
                                <a href="#" className="hover:text-white transition-colors"><Github className="w-5 h-5" /></a>
                            </li>
                        </ul>
                    </div>
                </div>
                <div className="border-t border-slate-800 mt-12 pt-8 text-center text-sm text-slate-500">
                    © {new Date().getFullYear()} Approv.io. All rights reserved.
                </div>
            </div>
        </footer>
    );
};

export default Footer;
