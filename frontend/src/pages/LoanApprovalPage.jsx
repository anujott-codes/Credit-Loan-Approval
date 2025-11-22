import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { DollarSign, CheckCircle, XCircle, AlertCircle, Loader2, ChevronRight } from 'lucide-react';
import { predictLoan } from '../services/api';
import clsx from 'clsx';

const LoanApprovalPage = () => {
    const [formData, setFormData] = useState({
        loan_amount: 50000,
        loan_term: 12,
        no_of_dependents: 1,
        gender: 'Male',
        age: 35,
        education: 'Yes',
        self_employed: 'No',
        annual_income: 600000,
        cibil_score: 750,
        residential_assets_value: 1000000,
        commercial_assets_value: 1000000,
        luxury_assets_value: 1000000,
        bank_asset_value: 1000000,
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
            const response = await predictLoan(formData);
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
                        <h1 className="text-4xl font-bold text-slate-900 mb-4">Loan Application</h1>
                        <p className="text-slate-600">Fill in the details below to check your loan eligibility instantly.</p>
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
                                    {/* Loan Details */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                                            <DollarSign className="w-5 h-5 mr-2 text-primary-600" />
                                            Loan Details
                                        </h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Loan Amount (₹)</label>
                                                <input
                                                    type="number"
                                                    name="loan_amount"
                                                    value={formData.loan_amount}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                    required
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Term (Months)</label>
                                                <input
                                                    type="number"
                                                    name="loan_term"
                                                    value={formData.loan_term}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                    required
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    {/* Personal Info */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-900 mb-4">Personal Information</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Annual Income (₹)</label>
                                                <input
                                                    type="number"
                                                    name="annual_income"
                                                    value={formData.annual_income}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                    required
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">CIBIL Score</label>
                                                <input
                                                    type="number"
                                                    name="cibil_score"
                                                    value={formData.cibil_score}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                    required
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Gender</label>
                                                <select
                                                    name="gender"
                                                    value={formData.gender}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Male">Male</option>
                                                    <option value="Female">Female</option>
                                                    <option value="Other">Other</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Age</label>
                                                <input
                                                    type="number"
                                                    name="age"
                                                    value={formData.age}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Graduated?</label>
                                                <select
                                                    name="education"
                                                    value={formData.education}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Yes">Yes</option>
                                                    <option value="No">No</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Self Employed?</label>
                                                <select
                                                    name="self_employed"
                                                    value={formData.self_employed}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                >
                                                    <option value="Yes">Yes</option>
                                                    <option value="No">No</option>
                                                </select>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Assets */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-900 mb-4">Assets Valuation (₹)</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Residential Assets</label>
                                                <input
                                                    type="number"
                                                    name="residential_assets_value"
                                                    value={formData.residential_assets_value}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Commercial Assets</label>
                                                <input
                                                    type="number"
                                                    name="commercial_assets_value"
                                                    value={formData.commercial_assets_value}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Luxury Assets</label>
                                                <input
                                                    type="number"
                                                    name="luxury_assets_value"
                                                    value={formData.luxury_assets_value}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-slate-700 mb-2">Bank Assets</label>
                                                <input
                                                    type="number"
                                                    name="bank_asset_value"
                                                    value={formData.bank_asset_value}
                                                    onChange={handleChange}
                                                    className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                                                />
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
                                            'Check Eligibility'
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

                                            {result.approved && (
                                                <div className="bg-white/60 rounded-lg p-4">
                                                    <p className="text-sm text-slate-500 mb-1">Estimated Monthly EMI</p>
                                                    <p className="text-2xl font-bold text-slate-900">₹{result.emi.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
                                                    <p className="text-xs text-slate-500 mt-1">@ 8% p.a. interest rate</p>
                                                </div>
                                            )}

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

export default LoanApprovalPage;
