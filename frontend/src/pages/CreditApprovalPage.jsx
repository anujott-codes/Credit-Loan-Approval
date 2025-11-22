import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CreditCard, CheckCircle, XCircle, AlertCircle, Loader2 } from 'lucide-react';
import { predictCredit } from '../services/api';
import clsx from 'clsx';

const CreditApprovalPage = () => {
    const [formData, setFormData] = useState({
        Gender: 'Male',
        Age: 30,
        Married: 'Married',
        Citizen: 'ByBirth',
        Employment: 'Yes',
        Industry: 'InformationTechnology',
        YearsEmployed: 2,
        Income: 500000,
        Debt: 0,
        Bank_Customer: 'Yes',
        PriorDefault: 'No',
        CreditScore: 750,
        DriversLicense: 'Yes',
    });

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        const { name, value, type } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'number' ? parseFloat(value) : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await predictCredit(formData);
            setResult(response);
        } catch (err) {
            setError('Failed to process application. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 py-12">
            <div className="container mx-auto px-6">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-bold text-slate-900 mb-4">Credit Card Application</h1>
                        <p className="text-slate-600">Apply for a premium credit card with instant approval decisions.</p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                        {/* Form Section */}
                        <div className="lg:col-span-3">
                            <motion.form
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="bg-white rounded-2xl shadow-xl border border-slate-100 p-8"
                                onSubmit={handleSubmit}
                            >
                                <div className="space-y-8">
                                    {/* Personal Info */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                                            <CreditCard className="w-5 h-5 mr-2 text-primary-600" />
                                            Personal Information
                                        </h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Gender</label>
                                                <select
                                                    name="Gender"
                                                    value={formData.Gender}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Male">Male</option>
                                                    <option value="Female">Female</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Age</label>
                                                <input
                                                    type="number"
                                                    name="Age"
                                                    value={formData.Age}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Marital Status</label>
                                                <select
                                                    name="Married"
                                                    value={formData.Married}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Married">Married</option>
                                                    <option value="Single/Divorced/etc">Single/Divorced/etc</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Citizenship</label>
                                                <select
                                                    name="Citizen"
                                                    value={formData.Citizen}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="ByBirth">By Birth</option>
                                                    <option value="ByOtherMeans">By Other Means</option>
                                                    <option value="Temporary">Temporary</option>
                                                </select>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Employment Info */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-900 mb-4">Employment Details</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Employed?</label>
                                                <select
                                                    name="Employment"
                                                    value={formData.Employment}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Yes">Yes</option>
                                                    <option value="No">No</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Industry</label>
                                                <select
                                                    name="Industry"
                                                    value={formData.Industry}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Industrials">Industrials</option>
                                                    <option value="Materials">Materials</option>
                                                    <option value="CommunicationServices">Communication Services</option>
                                                    <option value="Transport">Transport</option>
                                                    <option value="InformationTechnology">Information Technology</option>
                                                    <option value="Financials">Financials</option>
                                                    <option value="Energy">Energy</option>
                                                    <option value="Real Estate">Real Estate</option>
                                                    <option value="Utilities">Utilities</option>
                                                    <option value="ConsumerDiscretionary">Consumer Discretionary</option>
                                                    <option value="Education">Education</option>
                                                    <option value="ConsumerStaples">Consumer Staples</option>
                                                    <option value="Healthcare">Healthcare</option>
                                                    <option value="Research">Research</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Years Employed</label>
                                                <input
                                                    type="number"
                                                    name="YearsEmployed"
                                                    value={formData.YearsEmployed}
                                                    onChange={handleChange}
                                                    step="0.1"
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Annual Income (₹)</label>
                                                <input
                                                    type="number"
                                                    name="Income"
                                                    value={formData.Income}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    {/* Financial Info */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-900 mb-4">Financial History</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Outstanding Debt (₹)</label>
                                                <input
                                                    type="number"
                                                    name="Debt"
                                                    value={formData.Debt}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Credit Score</label>
                                                <input
                                                    type="number"
                                                    name="CreditScore"
                                                    value={formData.CreditScore}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Bank Customer?</label>
                                                <select
                                                    name="Bank_Customer"
                                                    value={formData.Bank_Customer}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Yes">Yes</option>
                                                    <option value="No">No</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Prior Default?</label>
                                                <select
                                                    name="PriorDefault"
                                                    value={formData.PriorDefault}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="No">No</option>
                                                    <option value="Yes">Yes</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Drivers License?</label>
                                                <select
                                                    name="DriversLicense"
                                                    value={formData.DriversLicense}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Yes">Yes</option>
                                                    <option value="No">No</option>
                                                </select>
                                            </div>
                                        </div>
                                    </div>

                                    <button
                                        type="submit"
                                        disabled={loading}
                                        className="w-full bg-primary-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-primary-700 transition-all shadow-lg shadow-primary-500/30 disabled:opacity-70 disabled:cursor-not-allowed flex justify-center items-center"
                                    >
                                        {loading ? (
                                            <>
                                                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                                                Processing...
                                            </>
                                        ) : (
                                            'Submit Application'
                                        )}
                                    </button>
                                </div>
                            </motion.form>
                        </div>

                        {/* Results Section */}
                        <div className="lg:col-span-2">
                            <AnimatePresence mode="wait">
                                {result && (
                                    <motion.div
                                        initial={{ opacity: 0, x: 20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: 20 }}
                                        className={clsx(
                                            "rounded-2xl shadow-xl border p-6 sticky top-24",
                                            result.approved ? "bg-green-50 border-green-100" : "bg-red-50 border-red-100"
                                        )}
                                    >
                                        <div className="flex items-center mb-4">
                                            {result.approved ? (
                                                <CheckCircle className="w-8 h-8 text-green-600 mr-3" />
                                            ) : (
                                                <XCircle className="w-8 h-8 text-red-600 mr-3" />
                                            )}
                                            <h2 className={clsx("text-2xl font-bold", result.approved ? "text-green-800" : "text-red-800")}>
                                                {result.approved ? "Approved" : "Rejected"}
                                            </h2>
                                        </div>

                                        <div className="space-y-4">
                                            <div className="bg-white/60 rounded-lg p-4">
                                                <p className="text-sm text-slate-500 mb-1">Confidence Score</p>
                                                <div className="flex items-center">
                                                    <div className="flex-grow h-2 bg-slate-200 rounded-full mr-3">
                                                        <div
                                                            className={clsx("h-full rounded-full", result.approved ? "bg-green-500" : "bg-red-500")}
                                                            style={{ width: `${result.confidence}%` }}
                                                        />
                                                    </div>
                                                    <span className="font-bold text-slate-700">{result.confidence.toFixed(1)}%</span>
                                                </div>
                                            </div>

                                            <div className="mt-6 pt-6 border-t border-slate-100">
                                                <h3 className="font-semibold text-slate-800 mb-4 flex items-center">
                                                    <span className="bg-slate-100 p-1 rounded mr-2">📊</span>
                                                    Key Influencing Factors
                                                </h3>
                                                <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                                                    <div className="space-y-3">
                                                        {result.top_features.map((feature, index) => {
                                                            const isPositive = feature.value > 0;
                                                            // Clean up feature name
                                                            const cleanName = feature.feature
                                                                .replace(/^(num__|cat__|bin__)/, '')
                                                                .replace(/_/g, ' ');

                                                            return (
                                                                <div key={index} className="flex items-center justify-between text-sm group w-full">
                                                                    <span className="text-slate-600 font-medium mr-3 flex-1" title={cleanName}>
                                                                        {cleanName}
                                                                    </span>
                                                                    <div className={clsx(
                                                                        "flex items-center px-2.5 py-1 rounded-full text-xs font-bold whitespace-nowrap flex-shrink-0",
                                                                        isPositive
                                                                            ? "bg-green-100 text-green-700 border border-green-200"
                                                                            : "bg-red-100 text-red-700 border border-red-200"
                                                                    )}>
                                                                        {isPositive ? "Favorable" : "Unfavorable"}
                                                                        <span className="ml-1 opacity-75">
                                                                            ({isPositive ? "+" : ""}{feature.value.toFixed(2)})
                                                                        </span>
                                                                    </div>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {!result && !loading && (
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 text-center text-slate-400">
                                    <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                    <p>Submit the form to see your approval results here.</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CreditApprovalPage;
