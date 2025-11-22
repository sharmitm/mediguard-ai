'use client'

import RiskScoreBadge from './RiskScoreBadge'

interface AgentCardProps {
  title: string
  icon: string
  data: any
  isLoading?: boolean
}

export default function AgentCard({ title, icon, data, isLoading }: AgentCardProps) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <span className="text-2xl">{icon}</span>
          <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        </div>
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <span className="text-2xl">{icon}</span>
          <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        </div>
      </div>

      <div className="space-y-4">
        {/* Risk Score */}
        {data.fraud_risk_score !== undefined && (
          <RiskScoreBadge score={data.fraud_risk_score} label="Fraud Risk" />
        )}
        {data.billing_fraud_score !== undefined && (
          <RiskScoreBadge score={data.billing_fraud_score} label="Billing Fraud Risk" />
        )}

        {/* Flags */}
        {data.identity_misuse_flag !== undefined && (
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Identity Misuse:</span>
            <span className={`px-2 py-1 rounded text-xs font-medium ${
              data.identity_misuse_flag 
                ? 'bg-danger/10 text-danger' 
                : 'bg-success/10 text-success'
            }`}>
              {data.identity_misuse_flag ? 'Detected' : 'Not Detected'}
            </span>
          </div>
        )}

        {data.discharge_ready !== undefined && (
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Discharge Ready:</span>
            <span className={`px-2 py-1 rounded text-xs font-medium ${
              data.discharge_ready 
                ? 'bg-success/10 text-success' 
                : 'bg-warning/10 text-warning'
            }`}>
              {data.discharge_ready ? 'Yes' : 'No'}
            </span>
          </div>
        )}

        {/* Reasons/Flags List */}
        {(data.reasons || data.billing_flags || data.blockers) && (
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2">
              {data.reasons ? 'Reasons:' : data.billing_flags ? 'Billing Flags:' : 'Blockers:'}
            </h4>
            <ul className="space-y-1">
              {(data.reasons || data.billing_flags || data.blockers || []).map((item: string, index: number) => (
                <li key={index} className="text-sm text-gray-600 flex items-start space-x-2">
                  <span className="text-primary mt-1">•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Explanations */}
        {data.explanation && (
          <div>
            <p className="text-sm text-gray-600">{data.explanation}</p>
          </div>
        )}

        {data.billing_explanation && (
          <div>
            <p className="text-sm text-gray-600">{data.billing_explanation}</p>
          </div>
        )}

        {/* Delay Hours */}
        {data.delay_hours !== undefined && (
          <div>
            <span className="text-sm font-medium text-gray-700">Estimated Delay: </span>
            <span className="text-sm text-gray-600">{data.delay_hours} hours</span>
          </div>
        )}
      </div>
    </div>
  )
}

